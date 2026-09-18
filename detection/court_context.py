"""Validated, resolution-independent court geometry and player context cues."""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import math

import numpy as np


def normalize_court_context(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Validate normalized net/zone coordinates without silently moving them.

    ``net_radius`` is a fraction of image width (distance calculations correct
    the y axis by the source aspect ratio).  Zones contain 3–12 vertices.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Court setup must be an object")

    def point(raw: Any) -> list[float]:
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise ValueError("Court points must have x and y coordinates")
        result = [float(x) for x in raw]
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in result):
            raise ValueError("Court coordinates must be between 0 and 1")
        return result

    net = point(value["net"]) if value.get("net") is not None else None
    radius = float(value.get("net_radius", 0.08))
    if not math.isfinite(radius) or not 0.005 <= radius <= 0.5:
        raise ValueError("Net radius must be between 0.005 and 0.5 of image width")
    raw_zones = value.get("serving_zones", [])
    if not isinstance(raw_zones, (tuple, list)) or len(raw_zones) > 4:
        raise ValueError("Choose at most four serving zones")
    zones = []
    for raw_zone in raw_zones:
        if not isinstance(raw_zone, (tuple, list)) or not 3 <= len(raw_zone) <= 12:
            raise ValueError("Each serving zone needs 3–12 corners")
        zone = [point(p) for p in raw_zone]
        area = abs(sum(zone[i][0] * zone[(i + 1) % len(zone)][1]
                       - zone[(i + 1) % len(zone)][0] * zone[i][1]
                       for i in range(len(zone)))) / 2
        if area < 0.0001:
            raise ValueError("Serving zones must enclose a visible area")
        zones.append(zone)
    return {"net": net, "net_radius": radius, "serving_zones": zones}


def point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    """Ray casting, including polygon boundaries."""
    x, y = point
    inside = False
    for index, (ax, ay) in enumerate(polygon):
        bx, by = polygon[(index + 1) % len(polygon)]
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if abs(cross) < 1e-9 and min(ax, bx) - 1e-9 <= x <= max(ax, bx) + 1e-9 and min(ay, by) - 1e-9 <= y <= max(ay, by) + 1e-9:
            return True
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


def player_context_features(
    players: Sequence[Mapping[str, Any]], court: Mapping[str, Any] | None,
    *, aspect_ratio: float = 16 / 9,
) -> dict[str, float]:
    """Soft cues from detected people, never a proof that an action occurred.

    No person detection produces no evidence, rather than evidence of no play.
    HOG boxes only yield weak collection evidence; pose bending can strengthen
    it when actual confident keypoints were supplied by a local model.
    """
    reliable = [p for p in players if float(p.get("confidence", 0)) >= 0.30]
    result = dict(player_count=float(len(reliable)), readiness=0.0,
                  player_motion=0.0, retrieval=0.0, pose_serve=0.0)
    if not reliable:
        return result
    speeds = np.asarray([float(p.get("speed", 0)) for p in reliable])
    observed = np.asarray([bool(p.get("motion_observed", True)) for p in reliable])
    result["player_motion"] = float(np.mean(np.clip(speeds / 0.10, 0, 1) * observed))
    if court is None:
        return result
    zones = court.get("serving_zones", [])
    in_zone = [any(point_in_polygon(p["foot"], zone) for zone in zones) for p in reliable]
    stable = float(np.mean(np.clip(1 - speeds / 0.045, 0, 1) * observed))
    if len(reliable) >= 2 and any(in_zone):
        points = np.asarray([p["foot"] for p in reliable])
        separation = float(np.max(np.linalg.norm((points[:, None] - points[None, :])
                                                 * [1, 1 / aspect_ratio], axis=2)))
        result["readiness"] = stable * min(1, separation / 0.20) * min(1, len(reliable) / 3)
    net = court.get("net")
    if net is not None:
        radius = float(court.get("net_radius", 0.08))
        near = [p for p in reliable if math.hypot(p["foot"][0] - net[0],
                                                 (p["foot"][1] - net[1]) / aspect_ratio) <= radius * 1.8]
        if near:
            near_slow = float(np.mean([max(0, 1 - float(p.get("speed", 0)) / 0.08)
                                       if p.get("motion_observed", True) else 0 for p in near]))
            bend = max(float(p.get("bend", 0)) for p in near)
            # Grouping at the net is ambiguous; the no-pose signal is limited.
            result["retrieval"] = min(1, (0.32 + 0.60 * bend) * near_slow * len(near) / len(reliable))
    result["pose_serve"] = max((float(p.get("arm_swing", 0)) for p, valid in zip(reliable, in_zone) if valid), default=0.0)
    return result


__all__ = ["normalize_court_context", "point_in_polygon", "player_context_features"]
