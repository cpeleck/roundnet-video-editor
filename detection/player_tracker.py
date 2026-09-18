"""Local human detection, short optical-flow tracks, and honest pose cues.

macOS uses the system Vision pose model through a small persistent Swift worker.
Other platforms use OpenCV's bundled HOG human classifier. Neither backend
claims ball tracking or action classification; pose/formation signals are soft
evidence to review, and uncertain/occluded people cannot veto a rally.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

from config import DetectionSettings
from .court_context import player_context_features


class VisionPose:
    def __init__(self, cv2: Any) -> None:
        self.cv2 = cv2
        source = Path(__file__).with_name("vision_pose.swift")
        cache = Path(tempfile.gettempdir()) / "roundnet-vision-pose"
        cache.mkdir(exist_ok=True)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        binary = cache / f"vision-pose-{digest}"
        if not binary.is_file():
            compiler = shutil.which("swiftc")
            if compiler is None:
                raise RuntimeError("Swift compiler is unavailable")
            # A unique output prevents concurrent analyses from executing a
            # binary while another process is still linking it.
            import uuid
            output = cache / f"build-{uuid.uuid4().hex}"
            result = subprocess.run(
                [compiler, str(source), "-O", "-module-cache-path", str(cache / "modules"),
                 "-o", str(output)], capture_output=True, text=True, timeout=90,
            )
            if result.returncode:
                raise RuntimeError("Could not compile macOS pose helper: " + result.stderr[-1000:])
            output.replace(binary)
        self.process = subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        try:
            ready = self._read(15)
            if not ready.get("ready"):
                raise RuntimeError("macOS pose helper did not initialize")
        except Exception:
            self.close()
            raise

    def _read(self, timeout: float = 15) -> dict[str, Any]:
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise RuntimeError("macOS pose analysis timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("macOS pose helper stopped unexpectedly")
        return json.loads(line)

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        ok, image = self.cv2.imencode(".jpg", frame, [self.cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return []
        request = json.dumps({"image": base64.b64encode(image).decode("ascii")})
        self.process.stdin.write(request + "\n")
        self.process.stdin.flush()
        payload = self._read()
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        people = []
        for points in payload.get("people", []):
            keypoints = np.asarray(points, dtype=float)
            valid = keypoints[:, 2] >= 0.30
            if np.count_nonzero(valid) < 5:
                continue
            xy = keypoints[valid, :2]
            left, top = np.maximum(0, np.min(xy, axis=0) - [0.025, 0.025])
            right, bottom = np.minimum(1, np.max(xy, axis=0) + [0.025, 0.025])
            people.append({"bbox": [float(left), float(top), float(right - left), float(bottom - top)],
                           "confidence": float(np.mean(keypoints[valid, 2])), "keypoints": keypoints})
        return people

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()


class HOGPeople:
    def __init__(self, cv2: Any) -> None:
        self.cv2 = cv2
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        height, width = frame.shape[:2]
        if height < 128 or width < 64:
            return []
        boxes, weights = self.hog.detectMultiScale(frame, winStride=(8, 8),
                                                  padding=(8, 8), scale=1.08)
        result = []
        for (x, y, w, h), weight in zip(boxes, np.asarray(weights).reshape(-1)):
            if weight < 0.4:
                continue
            result.append({"bbox": [x / width, y / height, w / width, h / height],
                           "confidence": min(1, 0.45 + float(weight) * 0.2)})
        # HOG's grouping leaves some nested duplicates. Keep highest confidence.
        kept = []
        for person in sorted(result, key=lambda p: -p["confidence"]):
            if all(_iou(person["bbox"], other["bbox"]) < 0.45 for other in kept):
                kept.append(person)
        return kept

    def close(self) -> None:
        pass


class _UnavailablePeople:
    """Optional person analysis must not prevent the motion detector working."""

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        return []

    def close(self) -> None:
        pass


class ONNXPose:
    """Optional fixed 640px YOLOv8/11 COCO17 ONNX, exported without NMS."""
    def __init__(self, cv2: Any, path: str) -> None:
        self.cv2 = cv2
        model = Path(path).expanduser()
        if not model.is_file():
            raise ValueError(f"Pose model not found: {model}")
        self.net = cv2.dnn.readNetFromONNX(str(model))

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        height, width = frame.shape[:2]
        scale = min(640 / width, 640 / height)
        resized = self.cv2.resize(frame, (round(width * scale), round(height * scale)))
        left, top = (640 - resized.shape[1]) // 2, (640 - resized.shape[0]) // 2
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
        self.net.setInput(self.cv2.dnn.blobFromImage(canvas, 1 / 255, (640, 640), swapRB=True))
        rows = np.asarray(self.net.forward()).squeeze()
        if rows.ndim != 2 or 56 not in rows.shape:
            raise ValueError("Pose ONNX must export COCO17 raw predictions shaped [1,56,N] (640px, no NMS)")
        if rows.shape[0] == 56:
            rows = rows.T
        result = []
        for row in rows[rows[:, 4] >= 0.40]:
            cx, cy, w, h = row[:4]
            x = (cx - w / 2 - left) / scale / width
            y = (cy - h / 2 - top) / scale / height
            points = row[5:].reshape(17, 3).astype(float).copy()
            points[:, 0] = (points[:, 0] - left) / scale / width
            points[:, 1] = (points[:, 1] - top) / scale / height
            result.append({"bbox": [float(x), float(y), float(w / scale / width), float(h / scale / height)],
                           "confidence": float(row[4]), "keypoints": points})
        kept = []
        for person in sorted(result, key=lambda p: -p["confidence"]):
            if all(_iou(person["bbox"], other["bbox"]) < 0.45 for other in kept):
                kept.append(person)
        return kept

    def close(self) -> None:
        pass


@dataclass
class _Track:
    id: int
    person: dict[str, Any]
    detected_at: float
    points: np.ndarray | None = None
    observed_bbox: list[float] | None = None


class PlayerTracker:
    def __init__(self, settings: DetectionSettings, cv2: Any, court: dict[str, Any] | None = None) -> None:
        self.settings, self.cv2, self.court = settings, cv2, court
        self.warnings: list[str] = []
        self.backend_name = "OpenCV HOG human detector + optical flow"
        try:
            if settings.pose_model_path:
                self.backend = ONNXPose(cv2, settings.pose_model_path)
                self.backend_name = "Local COCO17 ONNX pose + optical flow"
            elif sys.platform == "darwin":
                self.backend = VisionPose(cv2)
                self.backend_name = "macOS Vision body pose + optical flow"
            else:
                self.backend = HOGPeople(cv2)
        except Exception as exc:
            self.warnings.append(f"Pose unavailable: {exc}")
            self._fallback()
        self.tracks: list[_Track] = []
        self.next_id = 1
        self.last_detection = -math.inf
        self.previous_gray: np.ndarray | None = None
        self.previous_time: float | None = None

    def _fallback(self) -> None:
        try:
            self.backend = HOGPeople(self.cv2)
            self.backend_name = "OpenCV HOG human detector + optical flow"
            self.warnings.append("Using HOG person detection; body-pose cues are unavailable.")
        except Exception as exc:
            self.backend = _UnavailablePeople()
            self.backend_name = "unavailable (motion/audio analysis only)"
            self.warnings.append(f"Person detection unavailable; motion analysis continues: {exc}")

    def process(self, frame: np.ndarray, timestamp: float,
                roi: tuple[int, int, int, int] | None = None) -> tuple[dict[str, float], list[dict[str, Any]]]:
        height, width = frame.shape[:2]
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        dt = timestamp - self.previous_time if self.previous_time is not None else 0
        for track in self.tracks:
            # An earlier arm swing or speed measurement is not a fresh
            # observation when feature tracking fails on a later frame.
            track.person["speed"] = 0.0
            track.person["motion_observed"] = False
            track.person["arm_swing"] = 0.0
        if self.previous_gray is not None and dt > 0:
            for track in self.tracks:
                points = track.points
                if points is None or len(points) < 3:
                    continue
                moved, status, _ = self.cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, points, None,
                                                               winSize=(21, 21), maxLevel=3)
                if moved is None or status is None:
                    continue
                valid = status.reshape(-1) == 1
                if np.count_nonzero(valid) < 3:
                    continue
                shift = np.median((moved - points)[valid], axis=0).reshape(2) / [width, height]
                if np.linalg.norm(shift) > 0.15:
                    continue
                box = track.person["bbox"]
                box[0] = float(np.clip(box[0] + shift[0], 0, 1 - box[2]))
                box[1] = float(np.clip(box[1] + shift[1], 0, 1 - box[3]))
                track.person["speed"] = float(np.linalg.norm(shift * [1, height / width]) / dt)
                track.person["motion_observed"] = True
                track.points = moved[valid].reshape(-1, 1, 2)
                track.person["arm_swing"] = 0.0  # Only newly observed keypoints make a pose cue.
        if timestamp - self.last_detection >= self.settings.person_detection_interval - 1e-8:
            try:
                detected = self.backend.detect(frame)
            except Exception as exc:
                if isinstance(self.backend, HOGPeople):
                    self.backend.close()
                    self.backend = _UnavailablePeople()
                    self.backend_name = "unavailable (motion/audio analysis only)"
                    self.warnings.append(f"Person detection failed; motion analysis continues: {exc}")
                    detected = []
                else:
                    self.backend.close()
                    self.warnings.append(f"Pose analysis failed: {exc}")
                    self._fallback()
                    # Run the fallback at the next scheduled sample. That
                    # keeps a second backend failure inside this same guard.
                    detected = []
            used: set[int] = set()
            for candidate in detected:
                person = _validated_person(candidate)
                if person is None:
                    continue
                x, y, w, h = person["bbox"]
                foot = [x + w / 2, y + h]
                if roi is not None:
                    rx, ry, rw, rh = roi
                    if not rx <= foot[0] * width <= rx + rw or not ry <= foot[1] * height <= ry + rh:
                        continue
                matches = [( _match_score(person["bbox"], t.person["bbox"]), t)
                           for t in self.tracks if t.id not in used]
                score, track = max(matches, default=(0, None), key=lambda item: item[0])
                if score < 0.2:
                    track = None
                person["speed"] = 0.0 if track is None else track.person.get("speed", 0.0)
                person["motion_observed"] = bool(track and track.person.get("motion_observed", False))
                if track is not None and track.observed_bbox is not None and timestamp > track.detected_at:
                    previous_box = track.observed_bbox
                    previous_center = np.asarray([previous_box[0] + previous_box[2] / 2,
                                                  previous_box[1] + previous_box[3] / 2])
                    center = np.asarray([x + w / 2, y + h / 2])
                    person["speed"] = float(np.linalg.norm((center - previous_center)
                                                            * [1, height / width])
                                              / (timestamp - track.detected_at))
                    person["motion_observed"] = True
                person["bend"] = pose_bend(person.get("keypoints"))
                person["arm_swing"] = pose_arm_swing(
                    person.get("keypoints"), track.person.get("keypoints") if track else None,
                    timestamp - track.detected_at if track else 0,
                )
                if track is None:
                    track = _Track(self.next_id, person, timestamp)
                    self.next_id += 1
                    self.tracks.append(track)
                else:
                    track.person, track.detected_at = person, timestamp
                track.observed_bbox = person["bbox"].copy()
                used.add(track.id)
                mask = np.zeros_like(gray)
                left, top = max(0, int(x * width)), max(0, int(y * height))
                right, bottom = min(width, int((x + w) * width)), min(height, int((y + h) * height))
                mask[top:bottom, left:right] = 255
                track.points = self.cv2.goodFeaturesToTrack(gray, maxCorners=30, qualityLevel=0.01,
                                                            minDistance=5, mask=mask)
            self.last_detection = timestamp
        self.tracks = [t for t in self.tracks if timestamp - t.detected_at <= max(1.0, self.settings.person_detection_interval * 1.8)]
        players = []
        for track in self.tracks:
            person = dict(track.person)
            x, y, w, h = person["bbox"]
            person["foot"] = [x + w / 2, y + h]
            person["confidence"] *= max(0.0, 1 - (timestamp - track.detected_at) / 3)
            person["id"] = track.id
            players.append(person)
        features = player_context_features(players, self.court, aspect_ratio=width / height)
        self.previous_gray, self.previous_time = gray, timestamp
        serializable = [{key: value.tolist() if isinstance(value, np.ndarray) else value
                         for key, value in person.items()} for person in players]
        return features, serializable

    def close(self) -> None:
        self.backend.close()


def _validated_person(value: dict[str, Any]) -> dict[str, Any] | None:
    """Bound partially offscreen model boxes before optical-flow propagation."""
    try:
        box = np.asarray(value["bbox"], dtype=float)
        confidence = float(value["confidence"])
    except (KeyError, TypeError, ValueError):
        return None
    if box.shape != (4,) or not np.all(np.isfinite(box)) or not math.isfinite(confidence):
        return None
    if box[2] <= 0 or box[3] <= 0 or confidence < 0.30:
        return None
    left, top = np.clip(box[:2], 0, 1)
    right, bottom = np.clip(box[:2] + box[2:], 0, 1)
    if right <= left or bottom <= top:
        return None
    person = {"bbox": [float(left), float(top), float(right - left), float(bottom - top)],
              "confidence": min(1.0, confidence)}
    if value.get("keypoints") is not None:
        try:
            points = np.asarray(value["keypoints"], dtype=float)
            if points.shape == (17, 3) and np.all(np.isfinite(points)):
                person["keypoints"] = points.copy()
        except (TypeError, ValueError):
            pass
    return person


def pose_bend(points: np.ndarray | None) -> float:
    """Torso orientation cue from visible shoulders/hips; not action recognition."""
    if points is None:
        return 0.0
    if min(points[index, 2] for index in (5, 6, 11, 12)) < 0.35:
        return 0.0
    shoulder = np.mean(points[[5, 6], :2], axis=0)
    hip = np.mean(points[[11, 12], :2], axis=0)
    torso = shoulder - hip
    if np.linalg.norm(torso) < 0.02:
        return 0.0
    return float(np.clip((abs(torso[0]) / np.linalg.norm(torso) - 0.35) / 0.60, 0, 1))


def pose_arm_swing(current: np.ndarray | None, previous: np.ndarray | None, dt: float) -> float:
    """Wrist motion relative to shoulder, normalized by torso length and time."""
    if current is None or previous is None or not 0 < dt <= 1.5:
        return 0.0
    cues = []
    for shoulder, wrist, hip in ((5, 9, 11), (6, 10, 12)):
        if min(current[i, 2] for i in (shoulder, wrist, hip)) < 0.35 or min(previous[i, 2] for i in (shoulder, wrist)) < 0.35:
            continue
        scale = np.linalg.norm(current[shoulder, :2] - current[hip, :2])
        if scale < 0.03:
            continue
        movement = np.linalg.norm((current[wrist, :2] - current[shoulder, :2])
                                  - (previous[wrist, :2] - previous[shoulder, :2]))
        cues.append(float(np.clip((movement / scale / dt - 0.8) / 2.5, 0, 1)))
    return max(cues, default=0.0)


def _iou(a: list[float], b: list[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    return intersection / max(1e-9, a[2] * a[3] + b[2] * b[3] - intersection)


def _match_score(a: list[float], b: list[float]) -> float:
    distance = math.hypot(a[0] + a[2] / 2 - b[0] - b[2] / 2,
                          a[1] + a[3] / 2 - b[1] - b[3] / 2)
    return max(_iou(a, b), max(0, 0.5 - distance / 0.12))


__all__ = ["PlayerTracker", "VisionPose", "HOGPeople", "pose_bend", "pose_arm_swing"]
