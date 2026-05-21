"""Constants shared across the KWS pipeline."""

from __future__ import annotations

from typing import Iterable, List

DEFAULT_COMMANDS: List[str] = ["yes", "no", "up", "down", "left", "right", "stop", "go"]
SPECIAL_LABELS: List[str] = ["unknown", "silence"]
DEFAULT_LABELS: List[str] = DEFAULT_COMMANDS + SPECIAL_LABELS


def build_labels(commands: Iterable[str]) -> List[str]:
    command_list = list(commands)
    return command_list + SPECIAL_LABELS
