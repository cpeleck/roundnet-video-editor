from __future__ import annotations

import json

import numpy as np
import pytest

from config import DetectionSettings
from detection.court_context import normalize_court_context, player_context_features, point_in_polygon
from detection.player_tracker import PlayerTracker, pose_arm_swing, pose_bend
from detection.serve_detector import ServeDetector


def _court():
    return {"net": [0.5, 0.65], "net_radius": 0.08,
            "serving_zones": [[[0.1, 0.4], [0.3, 0.4], [0.3, 0.8], [0.1, 0.8]]]}


def _player(x, y, *, speed=0, bend=0, swing=0):
    return {"foot": [x, y], "confidence": 0.9, "speed": speed,
            "bend": bend, "arm_swing": swing}


def test_court_validation_rejects_invalid_coordinates_and_degenerate_zones():
    with pytest.raises(ValueError, match="between 0 and 1"):
        normalize_court_context({"net": [1.1, 0.5]})
    with pytest.raises(ValueError, match="visible area"):
        normalize_court_context({"serving_zones": [[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]]})
    context = normalize_court_context(_court())
    assert context == _court()
    assert point_in_polygon([0.2, 0.6], context["serving_zones"][0])
    assert not point_in_polygon([0.5, 0.6], context["serving_zones"][0])


def test_serving_formation_differs_from_gathering_at_net():
    formation = [_player(0.2, 0.65), _player(0.8, 0.65), _player(0.5, 0.25)]
    gathering = [_player(0.48, 0.65, bend=0.8), _player(0.52, 0.65, bend=0.8)]
    ready = player_context_features(formation, _court())
    retrieve = player_context_features(gathering, _court())
    assert ready["readiness"] > 0.9
    assert ready["retrieval"] == 0
    assert retrieve["readiness"] == 0
    assert retrieve["retrieval"] > 0.7


def test_missed_people_are_unknown_and_no_pose_is_only_weak_collection_evidence():
    unknown = player_context_features([], _court())
    assert all(value == 0 for value in unknown.values())
    gathering = player_context_features([_player(0.5, 0.65)], _court())
    assert 0 < gathering["retrieval"] <= 0.32
    assert gathering["pose_serve"] == 0


def test_pose_cues_require_visible_joints_and_relative_arm_motion():
    standing = np.zeros((17, 3))
    for i, xy in {5: (0.4, 0.3), 6: (0.5, 0.3), 11: (0.4, 0.5), 12: (0.5, 0.5),
                  9: (0.35, 0.5), 10: (0.55, 0.5)}.items():
        standing[i] = [*xy, 0.9]
    bent = standing.copy()
    bent[[5, 6], :2] += [0.2, 0.2]
    assert pose_bend(standing) == 0
    assert pose_bend(bent) > 0.9
    translated = standing.copy()
    translated[:, :2] += [0.1, 0.1]
    assert pose_arm_swing(translated, standing, 0.25) == pytest.approx(0)
    swing = standing.copy()
    swing[9, :2] += [0.2, -0.2]
    assert pose_arm_swing(swing, standing, 0.25) > 0.9
    swing[9, 2] = 0.1
    assert pose_arm_swing(swing, standing, 0.25) == 0


def test_player_tracking_preserves_identity_and_expires_unseen_tracks(monkeypatch):
    import cv2
    import detection.player_tracker as module

    class FakeBackend:
        def __init__(self, cv2):
            self.calls = 0

        def detect(self, frame):
            self.calls += 1
            if self.calls > 2:
                return []
            return [{"bbox": [0.2 + 0.01 * self.calls, 0.2, 0.1, 0.5], "confidence": 0.9}]

        def close(self):
            pass

    monkeypatch.setattr(module.sys, "platform", "test")
    monkeypatch.setattr(module, "HOGPeople", FakeBackend)
    tracker = PlayerTracker(DetectionSettings(person_detection_interval=0.5), cv2, _court())
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    _, first = tracker.process(frame, 0)
    _, second = tracker.process(frame, 0.5)
    assert first[0]["id"] == second[0]["id"]
    features, last = tracker.process(frame, 2.0)
    assert last == []
    assert features["player_count"] == 0
    tracker.close()


def test_context_cannot_turn_an_isolated_arm_swing_into_a_serve():
    timestamps = np.arange(0, 5, 0.125)
    zero = np.zeros_like(timestamps)
    one = np.ones_like(timestamps)
    result = ServeDetector().score(timestamps, zero, zero, zero,
                                   readiness=one, player_motion=zero, pose_serve=one)
    assert result.likelihood.max() < 0.1


def test_missing_context_preserves_original_serve_scores():
    timestamps = np.arange(0, 5, 0.125)
    active = ((timestamps > 2) & (timestamps < 4)).astype(float)
    detector = ServeDetector()
    baseline = detector.score(timestamps, active, active * 0.8, active * 0.1)
    zeros = np.zeros_like(active)
    unknown = detector.score(timestamps, active, active * 0.8, active * 0.1,
                              readiness=zeros, player_motion=zeros, retrieval=zeros, pose_serve=zeros)
    np.testing.assert_array_equal(baseline.likelihood, unknown.likelihood)


def test_native_or_fallback_tracker_is_not_required_for_legacy_results():
    from detection.detector import DetectionResult
    zeros = np.zeros(3)
    result = DetectionResult([], zeros, zeros, zeros, zeros, zeros, zeros, 1.0)
    assert result.player_count_scores.tolist() == [0, 0, 0]
    assert result.metadata == {}


def test_unknown_tracking_motion_is_not_stationary_formation_or_retrieval():
    players = [_player(0.2, 0.65), _player(0.5, 0.65, bend=1)]
    for player in players:
        player["motion_observed"] = False
    features = player_context_features(players, _court())
    assert features["readiness"] == 0
    assert features["retrieval"] == 0


def test_pose_startup_failure_falls_back_with_warning(monkeypatch):
    import cv2
    import detection.player_tracker as module

    def unavailable(cv2):
        raise RuntimeError("test model unavailable")

    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module, "VisionPose", unavailable)
    tracker = PlayerTracker(DetectionSettings(), cv2, _court())
    assert "HOG" in tracker.backend_name
    assert any("test model unavailable" in warning for warning in tracker.warnings)
    features, people = tracker.process(np.zeros((180, 320, 3), np.uint8), 0)
    assert features["player_count"] == 0
    json.dumps(people)
    tracker.close()


def test_all_people_backends_unavailable_does_not_break_motion_analysis(monkeypatch):
    import cv2
    import detection.player_tracker as module

    def unavailable(cv2):
        raise RuntimeError("test backend unavailable")

    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module, "VisionPose", unavailable)
    monkeypatch.setattr(module, "HOGPeople", unavailable)
    tracker = PlayerTracker(DetectionSettings(), cv2, _court())
    features, people = tracker.process(np.zeros((180, 320, 3), np.uint8), 0)
    assert people == []
    assert all(value == 0 for value in features.values())
    assert "motion/audio" in tracker.backend_name
    tracker.close()


def test_pose_inference_failure_switches_to_hog_without_failing_analysis(monkeypatch):
    import cv2
    import detection.player_tracker as module

    class FailedPose:
        closed = False

        def __init__(self, cv2):
            pass

        def detect(self, frame):
            raise RuntimeError("test inference failed")

        def close(self):
            self.closed = True

    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module, "VisionPose", FailedPose)
    tracker = PlayerTracker(DetectionSettings(), cv2, _court())
    pose = tracker.backend
    features, people = tracker.process(np.zeros((180, 320, 3), np.uint8), 0)
    assert people == []
    assert pose.closed
    assert "HOG" in tracker.backend_name
    assert any("test inference failed" in warning for warning in tracker.warnings)
    tracker.close()


def test_failed_hog_inference_disables_optional_analysis_once(monkeypatch):
    import cv2
    import detection.player_tracker as module

    class FailedHOG:
        def __init__(self, cv2):
            pass

        def detect(self, frame):
            raise RuntimeError("test HOG failed")

        def close(self):
            pass

    monkeypatch.setattr(module.sys, "platform", "test")
    monkeypatch.setattr(module, "HOGPeople", FailedHOG)
    tracker = PlayerTracker(DetectionSettings(), cv2, _court())
    frame = np.zeros((180, 320, 3), np.uint8)
    tracker.process(frame, 0)
    tracker.process(frame, 1)
    assert len(tracker.warnings) == 1
    assert "motion/audio" in tracker.backend_name
    tracker.close()


def test_invalid_and_offscreen_boxes_are_safe_and_serializable(monkeypatch):
    import cv2
    import detection.player_tracker as module

    class SuppliedPeople:
        def __init__(self, cv2):
            pass

        def detect(self, frame):
            return [{"bbox": [-0.1, -0.2, 0.4, 1.5], "confidence": np.float64(0.9),
                     "keypoints": np.zeros((17, 3))},
                    {"bbox": [float("nan"), 0, 0.2, 0.5], "confidence": 0.8},
                    {"bbox": [1.2, 0, 0.2, 0.5], "confidence": 0.8},
                    {"bbox": [0.1, 0.2, -0.2, 0.5], "confidence": 0.8}]

        def close(self):
            pass

    monkeypatch.setattr(module.sys, "platform", "test")
    monkeypatch.setattr(module, "HOGPeople", SuppliedPeople)
    tracker = PlayerTracker(DetectionSettings(), cv2, _court())
    features, people = tracker.process(np.zeros((180, 320, 3), np.uint8), 0)
    assert len(people) == 1
    assert people[0]["bbox"] == pytest.approx([0, 0, 0.3, 1])
    json.dumps({"features": features, "players": people}, allow_nan=False)
    tracker.close()


def test_tracker_does_not_repeat_pose_event_after_flow_failure(monkeypatch):
    import cv2
    import detection.player_tracker as module

    class SuppliedPeople:
        def __init__(self, cv2):
            pass

        def detect(self, frame):
            return [{"bbox": [0.1, 0.2, 0.2, 0.5], "confidence": 0.9}]

        def close(self):
            pass

    monkeypatch.setattr(module.sys, "platform", "test")
    monkeypatch.setattr(module, "HOGPeople", SuppliedPeople)
    tracker = PlayerTracker(DetectionSettings(), cv2, _court())
    frame = np.zeros((180, 320, 3), np.uint8)
    tracker.process(frame, 0)
    tracker.tracks[0].person.update(arm_swing=1.0, speed=0.1, motion_observed=True)
    _, people = tracker.process(frame, 0.125)
    assert people[0]["arm_swing"] == 0
    assert people[0]["speed"] == 0
    assert people[0]["motion_observed"] is False
    tracker.close()


def test_court_settings_and_detector_metadata_round_trip_json():
    from detection.detector import DetectionResult
    zeros = np.zeros(3)
    settings = DetectionSettings(pose_model_path="/tmp/local-pose.onnx", court_context_strength=0.2)
    assert DetectionSettings.from_dict(json.loads(json.dumps(settings.to_dict()))) == settings
    result = DetectionResult([], zeros, zeros, zeros, zeros, zeros, zeros, 1.0,
                             metadata={"court_context": normalize_court_context(_court()),
                                       "player_backend": "test", "player_tracks": []})
    assert json.loads(json.dumps(result.to_dict(include_signals=True)))["metadata"]["court_context"] == _court()
