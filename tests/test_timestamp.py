"""Small regression tests for the seconds-based UI timestamp boundary."""

from __future__ import annotations

import pytest

from ui.video_player import format_timestamp


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00.0"),
        (-4.0, "00:00.0"),
        (62.34, "01:02.3"),
        (3661.9, "1:01:01.9"),
    ],
)
def test_format_timestamp_uses_seconds_and_tenths(
    seconds: float, expected: str
) -> None:
    assert format_timestamp(seconds) == expected


def test_format_timestamp_can_hide_fractional_seconds() -> None:
    assert format_timestamp(125.9, tenths=False) == "02:05"

