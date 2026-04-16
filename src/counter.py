"""
Подсчёт дефектов с дедупликацией через track_id.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class DefectCounter:
    class_names: list[str]
    seen_track_ids: set[int] = field(default_factory=set)
    class_counts: Counter = field(default_factory=Counter)
    total_frames: int = 0

    def update(self, track_ids: list[int], class_ids: list[int]) -> None:
        self.total_frames += 1
        for tid, cid in zip(track_ids, class_ids):
            if tid is None or tid < 0:
                # без трекинга — не считаем, чтобы избежать дублей
                continue
            if tid in self.seen_track_ids:
                continue
            self.seen_track_ids.add(tid)
            self.class_counts[self.class_names[cid]] += 1

    def reset(self) -> None:
        self.seen_track_ids.clear()
        self.class_counts.clear()
        self.total_frames = 0

    def as_lines(self) -> list[str]:
        lines = [f"Frames: {self.total_frames}", f"Unique bottles: {len(self.seen_track_ids)}"]
        for cname in self.class_names:
            lines.append(f"{cname}: {self.class_counts[cname]}")
        return lines
