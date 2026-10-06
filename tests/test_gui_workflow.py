"""Opt-in native Qt integration: ROUNDNET_GUI_TESTS=1 python -m pytest -q ..."""

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("ROUNDNET_GUI_TESTS") != "1", reason="requires native desktop session")


@pytest.fixture
def window(tmp_path, monkeypatch, native_app):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication, QMessageBox
    import ui.main_window as module
    from ui.settings_dialog import DEFAULT_SETTINGS

    app = native_app
    settings = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(module, "QSettings", lambda *_: settings)
    monkeypatch.setattr(module.MainWindow, "_recover_last_project", lambda *_: None)
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: pytest.fail(str(args[-1])))
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: pytest.fail(str(args[-1])))
    editor = module.MainWindow()
    editor.detection_settings = deepcopy(DEFAULT_SETTINGS)
    editor._recovery_dir = tmp_path / "recovery"
    editor.show()
    editor.activateWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        app.setActiveWindow(editor)
    app.processEvents()
    source = Path("/private/tmp/roundnet_two_rallies.mp4")
    if not source.is_file():
        import shutil
        import subprocess
        if not shutil.which("ffmpeg"):
            pytest.skip("FFmpeg required")
        source = tmp_path / "source.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=320x180:rate=20:duration=15", "-c:v", "libx264", str(source)], check=True)
    editor.load_video(str(source), restore_recovery=False)
    from PySide6.QtTest import QTest
    assert editor.player.player.isAvailable(), "Qt multimedia backend is unavailable"
    for _ in range(100):
        app.processEvents()
        if editor.player.duration_seconds > 0:
            break
        QTest.qWait(20)
    assert editor.player.duration_seconds > 0
    yield editor, app
    editor.close()
    editor.deleteLater()
    app.processEvents()


@pytest.fixture(scope="session")
def native_app():
    # Qt Multimedia's process-wide integration must outlive every test window.
    import ui
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_native_review_recovery_and_annotation_workflow(window, tmp_path):
    from models import Rally
    from models.project import read_project, write_project
    from PySide6.QtWidgets import QTabWidget

    editor, app = window
    editor.rallies = [Rally(1, 3, .9, serve_confidence=.6), Rally(5, 8, .3)]
    editor.selected_rally_index = 0
    editor._rebuild_rally_tree()
    editor._update_summary()
    editor._update_ui_state()
    editor.toggle_star()
    assert editor.rallies[0].starred
    editor.undo_edit()
    assert not editor.rallies[0].starred
    editor.redo_edit()
    assert editor.rallies[0].starred
    editor.delete_selected_rally()
    assert editor.rallies[0].rejected and not editor.rallies[0].enabled
    editor.restore_rally()
    assert editor.rallies[0].enabled and not editor.rejected_detections
    editor.winner_combo.setCurrentIndex(1)
    from models.point_stats import normalize_roster
    roster = normalize_roster()
    roster[0]["name"] = "Alex"
    editor.match_settings["players"] = roster
    editor.outcome_combo.setCurrentText("Ace")
    editor.player_edit.setText("Alex")
    editor.note_edit.setText("Well placed")
    editor.apply_point_tag()
    assert editor.rallies[0].winner == "A"
    assert editor.rallies[0].point_stats["events"][0]["result"] == "ace"
    from models.statistics import calculate_statistics
    assert calculate_statistics(editor.rallies, editor.match_settings)["players"][0]["aces"] == 1
    assert "1 : 0" in editor.match_score_label.text()
    editor.next_uncertain()
    assert editor.selected_rally_index == 1
    editor.review_filter.setCurrentIndex(0)
    editor.complete_review_checkbox.setChecked(True)
    state = editor._project_snapshot()
    project = write_project(tmp_path / "game.roundnet.json", state)
    editor._apply_project(read_project(project))
    assert editor.rallies[0].starred and editor.complete_review_checkbox.isChecked()
    assert not editor._project_dirty and not editor._history.undo_stack
    editor._select_rally(0, seek=False)
    editor.findChild(QTabWidget).setCurrentIndex(1)
    assert editor.rally_tree.height() >= 140
    editor.note_edit.setFocus()
    app.processEvents()
    assert not any(s.isEnabled() for s in editor._shortcuts)
    editor.rally_tree.setFocus()
    app.processEvents()
    assert all(s.isEnabled() for s in editor._shortcuts)
    editor.grab().save(str(tmp_path / "editor.png"))


def test_native_analysis_undo_preserves_feature_provenance(window):
    from models import Rally
    editor, _ = window
    timestamps = np.arange(0, 10, .5)
    first = SimpleNamespace(rallies=[Rally(1, 3)], timestamps=timestamps,
                            rally_scores=np.zeros_like(timestamps), duration=10, warnings=())
    second = SimpleNamespace(rallies=[Rally(5, 8)], timestamps=timestamps,
                             rally_scores=np.ones_like(timestamps), duration=10, warnings=())
    editor._on_detection_succeeded(first)
    editor._on_detection_succeeded(second)
    editor.undo_edit()
    assert editor.rallies[0].start_time == 1
    assert np.all(editor.detection_result.rally_scores == 0)
    editor.redo_edit()
    assert editor.rallies[0].start_time == 5
    assert np.all(editor.detection_result.rally_scores == 1)
    editor.complete_review_checkbox.setChecked(True)
    editor.toggle_star()
    assert editor.complete_review_checkbox.isChecked()


def test_native_dialogs_and_preview(window):
    from models import Rally
    from ui.crop_editor import CropEditorDialog
    from ui.court_setup import CourtSetupDialog
    from ui.main_window import ExportDialog
    from ui.settings_dialog import SettingsDialog
    from ui.learning_dialog import LearningDialog
    editor, app = window
    rally = Rally(1, 3, starred=True, crop_keyframes=[
        {"time": 1, "x": .2, "y": .5}, {"time": 3, "x": .8, "y": .5}])
    crop = CropEditorDialog(editor.video_path, rally, editor)
    crop.load_frame(2)
    assert crop.canvas.center == pytest.approx((.5, .5))
    crop.reject()
    court = CourtSetupDialog(editor.video_path, None, editor)
    court.reject()
    learn = LearningDialog(editor)
    assert learn.profile_path is None
    learn.reject()
    settings = SettingsDialog(editor.detection_settings, editor)
    assert settings.values()["player_tracking_enabled"] is True
    assert settings.values()["pose_model_path"] == ""
    settings.reject()
    export = ExportDialog("/private/tmp/smoke-result.mp4", False, False, 1, 2, editor)
    export.aspect_combo.setCurrentIndex(2)
    export.scoreboard_checkbox.setChecked(True)
    export.stats_checkbox.setChecked(True)
    export.stats_duration.setValue(6.5)
    assert export.stats_preview.pixmap() is not None
    assert export.stats_duration.isEnabled()
    assert export.export_options["aspect_ratio"] == "9:16"
    assert export.export_options["scoreboard"] is True
    assert export.export_options["stats_duration"] == 6.5
    export.reject()
    editor.player.preview_ranges([(1, 2), (3, 4)])
    editor.player.pause()
    assert editor.player._preview_ranges == [(1000, 2000), (3000, 4000)]
    editor.player.player.setPosition(2000)
    editor.player._check_preview_boundary()
    assert editor.player._preview_index == 1
    editor.player.set_position(0)
    assert editor.player._preview_ranges == []
    app.processEvents()


def test_native_project_switch_cancel_and_source_mismatch(window, tmp_path, monkeypatch):
    from models import Rally
    from models.project import write_project
    from PySide6.QtWidgets import QMessageBox
    editor, _ = window
    editor.rallies = [Rally(1, 3, reviewed=True)]
    editor.complete_review_checkbox.setChecked(True)
    saved = editor._project_snapshot()
    saved["signals"] = {"timestamps": [0, 1], "rally_scores": [0, 1]}
    project = write_project(tmp_path / "source.roundnet.json", saved)
    # A refused switch must not apply a different saved edit on the same source.
    editor.rallies = [Rally(5, 8)]
    monkeypatch.setattr(editor, "_ensure_saved_before_leaving", lambda: False)
    editor._open_project_path(project, explicit=True)
    assert editor.rallies[0].start_time == 5
    monkeypatch.setattr(editor, "_ensure_saved_before_leaving", lambda: True)
    payload = json.loads(project.read_text())
    payload["source_stat"]["size"] += 1
    project.write_text(json.dumps(payload))
    editor._open_project_path(project, explicit=True)
    assert editor.rallies[0].start_time == 1
    assert editor.detection_result is None
    assert not editor.rallies[0].reviewed
    assert not editor.complete_review_checkbox.isChecked()
    assert editor.initial_detected_rallies == []
    assert editor.rejected_detections == []


def test_native_autosave_failure_retains_dirty_work(window, monkeypatch):
    import ui.project_workflow as module
    editor, _ = window
    original_save = module.write_project
    def fail(*args):
        raise OSError("Simulated full disk")
    monkeypatch.setattr(module, "write_project", fail)
    editor._project_dirty = True
    assert editor._autosave() is False
    assert editor._project_dirty
    assert "Autosave failed" in editor.statusBar().currentMessage()
    monkeypatch.setattr(module, "write_project", original_save)
    assert editor._autosave() is True
    assert not editor._project_dirty


def test_native_background_export_with_presentation(window, tmp_path):
    from models import Rally
    from PySide6.QtCore import QEventLoop, QTimer
    from ui.workers import ExportWorker
    from video.metadata import probe_video_metadata

    editor, _ = window
    output = tmp_path / "portrait-highlights.mp4"
    rallies = [Rally(0, .5, enabled=False, winner="A"),
               Rally(1, 2, starred=True, winner="B", note="Nice play", player="Alex")]
    worker = ExportWorker(editor.video_path, str(output), rallies, False)
    worker.export_options = {"aspect_ratio": "9:16", "highlights_only": True,
                                 "scoreboard": True, "include_stats": True, "include_notes": True,
                                 "stats_duration": 4.0,
                                 "team_a": "Équipe A", "team_b": "Team B"}
    finished, errors = [], []
    loop = QEventLoop()
    worker.succeeded.connect(finished.append)
    worker.failed.connect(errors.append)
    worker.finished.connect(loop.quit)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(30000)
    worker.start()
    loop.exec()
    timer.stop()
    if worker.isRunning():
        worker.request_cancel()
        worker.wait(5000)
    assert not worker.isRunning()
    assert not errors, errors
    assert finished
    metadata = probe_video_metadata(output)
    width, height = metadata.display_resolution
    assert width / height == pytest.approx(9/16)
    assert metadata.duration == pytest.approx(5, abs=.15)


def _logged_point(start=1, server="A1", receiver="B1"):
    from models import Rally
    from models.point_stats import new_event
    partner = lambda pid: pid[0] + ("2" if pid[1] == "1" else "1")
    events = [new_event(server, "serve", "in"), new_event(receiver, "receive", "strong"),
              new_event(partner(receiver), "set", "strong"), new_event(receiver, "hit", "auto"),
              new_event(partner(server), "defense", "auto"), new_event(server, "set", "strong"),
              new_event(partner(server), "hit", "auto")]
    return Rally(start, start+2, winner=server[0], point_stats={"version": 1, "complete": True,
        "server_id": server, "receiver_id": receiver, "events": events})


def test_native_touch_editor_saves_ordered_events_and_undo(window, tmp_path, monkeypatch):
    from models import Rally
    from models.statistics import calculate_statistics
    from ui.point_statistics import PointStatisticsDialog

    editor, app = window
    editor.rallies = [Rally(1, 10)]
    editor._rebuild_rally_tree(0, seek=False)
    editor._update_ui_state()
    expected = _logged_point().point_stats

    def edit(dialog):
        dialog.show()
        from PySide6.QtTest import QTest
        dialog.video.play()
        for _ in range(100):
            app.processEvents()
            if dialog.video.video_widget.videoSink().videoFrame().isValid():
                break
            QTest.qWait(20)
        dialog.video.pause()
        assert dialog.video.video_widget.videoSink().videoFrame().isValid()
        dialog.server.setCurrentIndex(dialog.server.findData("A2"))
        assert dialog.actor.currentData() == "A2"
        dialog.server.setCurrentIndex(dialog.server.findData("A1"))
        dialog.receiver.setCurrentIndex(dialog.receiver.findData("B1"))
        for event in expected["events"]:
            dialog.kind.setCurrentIndex(dialog.kind.findData(event["kind"]))
            dialog.actor.setCurrentIndex(dialog.actor.findData(event["player_id"]))
            dialog.result_combo.setCurrentIndex(dialog.result_combo.findData(event["result"]))
            dialog.add_event()
        assert dialog.table.currentRow() == -1
        assert len(dialog.events) == 7
        dialog.table.selectRow(5)
        identity = dialog.events[5]["event_id"]
        dialog.result_combo.setCurrentIndex(dialog.result_combo.findData("weak"))
        dialog.update_event()
        assert dialog.events[5]["event_id"] == identity
        assert dialog.events[5]["result"] == "weak"
        dialog.result_combo.setCurrentIndex(dialog.result_combo.findData("strong"))
        dialog.update_event()
        dialog.table.selectRow(4)
        dialog.move_event(-1)
        assert "Needs correction" in dialog.status.text()
        dialog.move_event(1)
        dialog.table.selectRow(6)
        dialog.remove_event()
        assert len(dialog.events) == 6
        dialog.kind.setCurrentIndex(dialog.kind.findData("hit"))
        dialog.actor.setCurrentIndex(dialog.actor.findData("A2"))
        dialog.result_combo.setCurrentIndex(dialog.result_combo.findData("auto"))
        dialog.add_event()
        dialog.winner_combo.setCurrentIndex(dialog.winner_combo.findData("A"))
        dialog.complete.setChecked(True)
        assert dialog.status.text().startswith("Complete point")
        assert "Get" in dialog.status.text() and "Put-away" in dialog.status.text()
        app.processEvents()
        assert dialog.grab().save(str(tmp_path / "touch-editor.png"))
        dialog._save()
        assert dialog.video.loaded_path is None
        return dialog.result()

    # Preserve the real save/accept path without starting a nested modal loop.
    monkeypatch.setattr(PointStatisticsDialog, "exec", edit)
    editor.edit_point_statistics()
    assert editor.rallies[0].point_stats["complete"]
    assert editor.rallies[0].winner == "A"
    report = calculate_statistics(editor.rallies)
    assert report["coverage"]["complete"] == 1
    assert report["players"][1]["defensive_gets"] == 1
    editor.undo_edit()
    assert editor.rallies[0].point_stats == {}
    editor.redo_edit()
    assert editor.rallies[0].point_stats["complete"]


def test_guided_ace_updates_score_stats_and_undo(window, monkeypatch):
    from models import Rally
    from models.statistics import calculate_statistics
    from ui.point_statistics import PointStatisticsDialog

    editor, _ = window
    editor.rallies = [Rally(1, 3)]
    editor._rebuild_rally_tree(0, seek=False)

    def log_ace(dialog):
        dialog.quick_touch("A1")
        assert dialog.events[0]["kind"] == "serve"
        dialog.quick_ace()
        assert dialog.complete.isChecked() and dialog.winner_combo.currentData() == "A"
        assert "1 : 0" in dialog.score_preview.text()
        dialog.quick_undo()
        assert dialog.events[0]["result"] == "in" and not dialog.complete.isChecked()
        dialog.quick_ace()
        dialog._save()
        return dialog.result()

    monkeypatch.setattr(PointStatisticsDialog, "exec", log_ace)
    editor.edit_point_statistics()
    assert "1 : 0" in editor.match_score_label.text()
    report = calculate_statistics(editor.rallies, editor.match_settings)
    assert report["players"][0]["aces"] == 1
    assert report["players"][2]["aced"] == 1
    assert report["coverage"]["complete"] == 1
    editor.undo_edit()
    assert editor.rallies[0].winner == "" and editor.rallies[0].point_stats == {}


def test_guided_fault_error_and_point_won(window):
    from models import Rally
    from models.statistics import point_issues
    from ui.point_statistics import PointStatisticsDialog

    editor, _ = window
    cases = [
        (("A1",), "quick_fault", "B", "fault"),
        (("A1", "B1", "B2"), "quick_error", "A", "error"),
        (("A1", "B1", "B2", "B1"), "quick_point_won", "B", "auto"),
    ]
    for players, action, winner, ending in cases:
        dialog = PointStatisticsDialog(editor.video_path, Rally(1, 3), editor.match_settings, editor)
        for player in players:
            dialog.quick_touch(player)
        getattr(dialog, action)()
        assert dialog.complete.isChecked()
        assert dialog.winner_combo.currentData() == winner
        assert dialog.events[-1]["result"] == ending
        assert point_issues(dialog._current_point(), winner) == []
        dialog.video.unload()
        dialog.deleteLater()


def test_native_statistics_tabs_exports_confirmation_and_point_navigation(window, tmp_path, monkeypatch):
    from models.point_stats import normalize_roster
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QFileDialog, QDialog
    from ui.statistics_dialog import StatisticsDialog

    editor, app = window
    editor.rallies = [_logged_point(1, "A1", "B1"), _logged_point(4, "A2", "B2"),
                      _logged_point(7, "B1", "A1"), _logged_point(10, "B2", "A2")]
    roster = normalize_roster()
    for p, name in zip(roster, ("Christian", "Alex", "Morgan", "Jordan")):
        p["name"] = name
    editor.match_settings["players"] = roster
    editor._rebuild_rally_tree(0, seek=False)
    editor._update_summary()

    def review(dialog):
        dialog.show()
        assert dialog.tabs.count() == 8
        assert dialog.report["coverage"]["complete"] == 4
        assert dialog.report["players"][0]["rpr"]["overall"] is None
        dialog.complete.setChecked(True)
        assert all(p["rpr"]["overall"] is not None for p in dialog.report["players"])
        app.processEvents()
        assert dialog.grab().save(str(tmp_path / "statistics-dialog.png"))
        for ext in (".png", ".csv", ".json"):
            destination = tmp_path / f"match-statistics{ext}"
            monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_, p=destination: (str(p), ""))
            dialog._export(ext)
            assert destination.is_file()
        card = QImage(str(tmp_path / "match-statistics.png"))
        assert card.width() == 1800 and card.height() == 850
        exported = json.loads((tmp_path / "match-statistics.json").read_text())
        assert exported["players"][0]["name"] == "Christian"
        assert exported["points"][0]["events"][-1]["result"] == "put_away"
        dialog._open_point(2, 0)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(StatisticsDialog, "exec", review)
    editor.show_match_statistics()
    assert editor.selected_rally_index == 2
    assert editor.match_settings["stats_complete"] is True
    editor.undo_edit()
    assert not editor.match_settings.get("stats_complete", False)
    editor.redo_edit()
    assert editor.match_settings["stats_complete"] is True
    editor.toggle_star()
    assert editor.match_settings["stats_complete"] is True


def test_native_touch_logs_survive_save_rejection_split_and_merge_undo(window, tmp_path):
    from models.project import read_project, write_project
    from models.statistics import calculate_statistics

    editor, app = window
    original = _logged_point()
    editor.rallies = [original, _logged_point(4)]
    editor._rebuild_rally_tree(0, seek=False)
    editor._update_summary()
    saved = write_project(tmp_path / "annotated.roundnet.json", editor._project_snapshot())
    editor._apply_project(read_project(saved))
    assert editor.rallies[0].point_stats == original.point_stats
    editor._select_rally(0, seek=False)
    editor.delete_selected_rally()
    assert calculate_statistics(editor.rallies)["coverage"]["complete"] == 1
    editor.restore_rally()
    assert calculate_statistics(editor.rallies)["coverage"]["complete"] == 2
    editor.player.set_position(2)
    app.processEvents()
    editor.split_selected_rally()
    assert len(editor.rallies) == 3
    assert editor.rallies[0].point_stats == editor.rallies[1].point_stats == {}
    editor.undo_edit()
    assert len(editor.rallies) == 2 and editor.rallies[0].point_stats == original.point_stats
    editor._select_rally(0, seek=False)
    editor.merge_selected_rally()
    assert len(editor.rallies) == 1 and editor.rallies[0].point_stats == {}
    editor.undo_edit()
    assert len(editor.rallies) == 2 and editor.rallies[0].point_stats == original.point_stats
