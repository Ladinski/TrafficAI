# src/tracker.py

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

Box = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass
class CarTrack:
    id: int
    box: Box
    start_time: datetime
    last_seen_time: datetime
    missed_frames: int = 0

    @property
    def duration_seconds(self) -> float:
        return (self.last_seen_time - self.start_time).total_seconds()


class Tracker:
    def __init__(
        self,
        max_distance: float = 50.0,
        max_missed_frames: int = 10,
    ):
        """
        :param max_distance: max pixel distance between centroids to consider the same car
        :param max_missed_frames: how many frames a car can be missed before we mark it as gone
        """
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames

        self.next_id: int = 1
        self.active_tracks: Dict[int, CarTrack] = {}
        self.finished_tracks: List[CarTrack] = []

    @staticmethod
    def _centroid(box: Box) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return cx, cy

    @staticmethod
    def _euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def update(self, detections: List[dict], current_time: Optional[datetime] = None):
        """
        Update tracker with new detections for the current frame.

        :param detections: list of {"box": (x1, y1, x2, y2), "conf": float}
        :param current_time: datetime for this frame (if None, uses now)
        """
        if current_time is None:
            current_time = datetime.now()

        
        det_boxes: List[Box] = [d["box"] for d in detections]
        det_centroids = [self._centroid(box) for box in det_boxes]

        
        unmatched_detections = set(range(len(det_boxes)))
        unmatched_tracks = set(self.active_tracks.keys())

        
        for track_id, track in list(self.active_tracks.items()):
            track_centroid = self._centroid(track.box)

            best_det_idx = None
            best_distance = float("inf")

            for det_idx in list(unmatched_detections):
                det_centroid = det_centroids[det_idx]
                dist = self._euclidean_distance(track_centroid, det_centroid)

                if dist < best_distance:
                    best_distance = dist
                    best_det_idx = det_idx

            if best_det_idx is not None and best_distance <= self.max_distance:
               
                new_box = det_boxes[best_det_idx]
                track.box = new_box
                track.last_seen_time = current_time
                track.missed_frames = 0

                
                unmatched_detections.discard(best_det_idx)
                unmatched_tracks.discard(track_id)
            
       
        for det_idx in unmatched_detections:
            box = det_boxes[det_idx]
            new_track = CarTrack(
                id=self.next_id,
                box=box,
                start_time=current_time,
                last_seen_time=current_time,
                missed_frames=0,
            )
            self.active_tracks[self.next_id] = new_track
            self.next_id += 1

        
        for track_id in list(unmatched_tracks):
            track = self.active_tracks[track_id]
            track.missed_frames += 1

            if track.missed_frames > self.max_missed_frames:
                
                self.finished_tracks.append(track)
                del self.active_tracks[track_id]

    def get_finished_tracks(self, clear: bool = True) -> List[CarTrack]:
        """
        Return the list of finished tracks (cars that left the frame).
        :param clear: if True, clears the internal list after returning
        """
        finished = list(self.finished_tracks)
        if clear:
            self.finished_tracks.clear()
        return finished
