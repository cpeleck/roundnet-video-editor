"""Guided logging yields valid, durable RPR evidence without made-up grades."""
from copy import deepcopy

import pytest

from models.quick_stats import quick_point_stats
from models.statistics import calculate_statistics, point_issues, classified_point_issues
from models.rally import Rally
from tests.test_web_projects import service, new_project, add, edit


def log(*players, error=None, defender=None, faults=()):
    return dict(version=1, possessions=[dict(first_touch_id=p, hitter_id=p) for p in players],
                error=error, terminal_defender_id=defender, serve_faults=list(faults))


def classify(service, project, kind, details):
    return edit(service, project, 'classify', rally_id=project['rallies'][0]['rally_id'], kind=kind,
                quick_log=details, **({'player_id': details['error']['player_id']} if kind == 'error' else {}))


def test_atomic_classification_undo_and_restart(service, tmp_path):
    project = add(service, new_project(service, tmp_path), 2, 8)
    before = deepcopy(project)
    project = classify(service, project, 'defensive_hold', log('B1', 'A2', defender='B2', faults=['fault']))
    assert project['revision'] == before['revision'] + 1
    assert project['position'] == 8
    assert project['statistics']['coverage']['rpr_complete'] == 1
    assert project['statistics']['coverage']['touch_quality_complete'] == 0
    a1, a2, b1, b2 = project['statistics']['players']
    assert a1['serve_attempts'] == 2
    assert a2['defensive_gets'] == a2['put_aways'] == 1
    assert b1['hits_returned'] == 1
    assert b2['defensive_not_returned'] == 1
    assert all(p['strong_set_pct'] is None for p in project['statistics']['players'])
    restarted = type(service)(service.data_dir, metadata_probe=service.metadata_probe).snapshot(project['project_id'])
    assert restarted['rallies'] == project['rallies']
    undone = edit(service, project, 'undo')
    assert undone['rallies'] == before['rallies']
    assert edit(service, undone, 'redo')['rallies'] == project['rallies']


@pytest.mark.parametrize('kind,details', [
    ('defensive_hold', log('B1')),
    ('defensive_hold', log('B2', 'A1')),
    ('defensive_hold', log('B1', 'B2')),
    ('defensive_break', log('B1', 'A1')),
    ('defensive_hold', log('B1', 'A1', defender='A2')),
    ('error', log('B1', 'A1', error={'player_id': 'A1', 'kind': 'receive'})),
    ('error', log('B1', error={'player_id': 'A1', 'kind': 'hit'})),
    ('ace', log(faults=['fault', 'rim'])),
])
def test_bad_sequence_changes_nothing(service, tmp_path, kind, details):
    project = add(service, new_project(service, tmp_path), 2, 8)
    with pytest.raises(ValueError):
        classify(service, project, kind, details)
    assert service.snapshot(project['project_id']) == project


@pytest.mark.parametrize('kind,expected', [('ace', ('A1', 'aces')), ('double_fault', ('A1', 'double_faults')),
                                         ('service_break', ('B1', 'receive_errors')), ('sideout', ('B1', 'put_aways'))])
def test_no_rally_defaults_and_no_double_credit(service, tmp_path, kind, expected):
    project = classify(service, add(service, new_project(service, tmp_path), 0, 3), kind, log())
    assert not project['statistics']['warnings']
    player = next(p for p in project['statistics']['players'] if p['player_id'] == expected[0])
    assert player[expected[1]] == 1
    if kind == 'double_fault':
        assert player['faults'] == player['serve_attempts'] == 2
    assert all(p['defensive_touches'] == 0 for p in project['statistics']['players'])


@pytest.mark.parametrize('touch,culprit,possessions', [('receive', 'B1', ('B1',)), ('set', 'A2', ('B1', 'A1')),
                                                    ('hit', 'A1', ('B1', 'A1'))])
def test_error_type_and_prior_hitter(service, tmp_path, touch, culprit, possessions):
    details = log(*possessions, error={'player_id': culprit, 'kind': touch})
    project = classify(service, add(service, new_project(service, tmp_path), 0, 8), 'error', details)
    assert not project['statistics']['warnings']
    p = next(p for p in project['statistics']['players'] if p['player_id'] == culprit)
    assert p['errors'] == p[touch + '_errors'] == 1
    if len(possessions) > 1:
        assert project['statistics']['players'][2]['put_aways'] == 1


def test_rpr_quick_logs_without_touch_grades(service, tmp_path):
    project = classify(service, add(service, new_project(service, tmp_path), 0, 8), 'defensive_break', log('B1', 'A2', 'B2'))
    project = edit(service, project, 'full_match', confirmed=True)
    report = project['statistics']
    assert report['rpr_eligible']
    assert report['players'][2]['receive_pct'] is None
    assert report['players'][0]['rpr']['overall'] is None  # no hit denominator
    assert not report['warnings']


def test_corrected_intermediate_hitter_and_tough_set():
    details = log('B1', 'A1', 'B2')
    details['possessions'][0]['hitter_id'] = 'B2'
    details['possessions'][1]['tough_set'] = True
    assignment = dict(server_id='A1', receiver_id='B1', provisional=False, winner='B')
    classification = dict(version=1, kind='defensive_break')
    point = quick_point_stats(details, classification, assignment)
    assert not point_issues(point, 'B')
    assert not classified_point_issues(point, classification, assignment)
    assert any(e['player_id'] == 'B2' and e['kind'] == 'hit' for e in point['events'])
    report = calculate_statistics([Rally(0, 8, classification=classification, point_stats=point)], {'stats_complete': True})
    assert report['players'][1]['tough_touches'] == 1


def test_edit_quick_log_is_one_undo_and_same_log_is_noop(service, tmp_path):
    project = classify(service, add(service, new_project(service, tmp_path), 0, 8), 'defensive_hold', log('B1', 'A1'))
    original = deepcopy(project['rallies'])
    unchanged = classify(service, project, 'defensive_hold', log('B1', 'A1'))
    assert unchanged['revision'] == project['revision']
    updated = classify(service, project, 'defensive_hold', log('B1', 'A2', faults=['rim', 'let']))
    assert updated['statistics']['players'][0]['rims'] == 1
    assert updated['rallies'][0]['point_stats']['quick_log']['possessions'][1]['hitter_id'] == 'A2'
    assert edit(service, updated, 'undo')['rallies'] == original


def test_existing_full_log_is_preserved_when_reconfirming(service, tmp_path):
    from models.point_stats import new_event
    project = add(service, new_project(service, tmp_path), 0, 8)
    project = edit(service, project, 'classify', rally_id=project['rallies'][0]['rally_id'], kind='ace')
    full = dict(version=1, complete=True, server_id='A1', receiver_id='B1', events=[new_event('A1', 'serve', 'ace')])
    project = edit(service, project, 'touches', rally_id=project['rallies'][0]['rally_id'], point_stats=full)
    assert classify(service, project, 'ace', log())['rallies'][0]['point_stats'] == full


def test_scorecard_and_rpr_update_each_point_without_confirmation(service, tmp_path):
    from application.jobs import export_options
    from video.presentation import match_statistics
    project = new_project(service, tmp_path)
    for i in range(8):
        project = add(service, project, i * 5, i * 5 + 3)
    for i in range(7):
        identity = project['rallies'][i]['rally_id']
        project = edit(service, project, 'classify', rally_id=identity, kind='sideout', quick_log=log())
        report = project['statistics']
        assert not project['match_settings']['stats_complete']
        assert report['teams']['A']['score'] + report['teams']['B']['score'] == i + 1
        assert report['coverage']['rpr_complete'] == i + 1
        assert report['rpr_eligible'] and not report['rpr_final']
        assert report['players'][0]['serve_pct'] == 1
        summary = match_statistics(project['rallies'], export_options(service.get_state(project['project_id'])))
        assert summary['score_a'] + summary['score_b'] == i + 1
        assert summary['player_statistics'] == report
        assert report == calculate_statistics([Rally.from_dict(r) for r in project['rallies']], project['match_settings'])
    assert all(p['rpr']['overall'] == pytest.approx(55) for p in report['players'])
    # Editorial exclusions still contribute to the final match card.
    excluded = edit(service, project, 'export_enabled', rally_id=project['rallies'][0]['rally_id'], enabled=False)
    assert excluded['statistics']['players'] == report['players']
    assert excluded['statistics']['teams'] == report['teams']
    undone = edit(service, excluded, 'undo')
    undone = edit(service, undone, 'undo')
    assert undone['statistics']['coverage']['rpr_complete'] == 6
    restored = edit(service, undone, 'redo')
    assert restored['statistics'] == report
    restarted = type(service)(service.data_dir, metadata_probe=service.metadata_probe)
    assert restarted.snapshot(project['project_id'])['statistics'] == report


def test_future_unclassified_clips_do_not_change_running_rpr_or_rates():
    assignment = dict(server_id='A1', receiver_id='B1', winner='A', provisional=False)
    classification = dict(version=1, kind='defensive_hold')
    point = quick_point_stats(log('B1', 'A1'), classification, assignment)
    rally = Rally(0, 3, classification=classification, point_stats=point)
    recorded = calculate_statistics([rally])
    pending = calculate_statistics([rally, *[Rally(i * 5, i * 5 + 3) for i in range(1, 60)]])
    assert pending['rpr_eligible']
    assert pending['players'][0]['rpr']['overall'] == pytest.approx(63)
    assert pending['players'] == recorded['players']


@pytest.mark.parametrize('kind,details,full_log', [
    ('ace', log(), False),
    ('error', log('A2', error={'player_id': 'A2', 'kind': 'receive'}), False),
    ('ace', log(), True),
])
def test_reopening_repairs_generated_roles_and_preserves_observed_players(service, tmp_path, kind, details, full_log):
    from models.project import write_project
    project = add(service, new_project(service, tmp_path), 0, 3)
    project = add(service, project, 5, 8)
    first, second = project['rallies']
    project = edit(service, project, 'classify', rally_id=first['rally_id'], kind='ace', quick_log=log())
    project = edit(service, project, 'classify', rally_id=second['rally_id'], kind=kind, quick_log=details,
                   **({'player_id': 'A2'} if kind == 'error' else {}))
    state = service.get_state(project['project_id'])
    old_point = state['rallies'][1]['point_stats']
    old_point['server_id'] = 'B1'
    for event in old_point['events']:
        if event['kind'] == 'serve':
            event['player_id'] = 'B1'
    if full_log:
        old_point.pop('quick_log')
    original = deepcopy(old_point)
    write_project(service.project_dir / f"{project['project_id']}.roundnet.json", state)
    restarted = type(service)(service.data_dir, metadata_probe=service.metadata_probe)
    reopened = restarted.snapshot(project['project_id'])
    refreshed = reopened['rallies'][1]['point_stats']
    if full_log:
        assert refreshed == original
        assert reopened['statistics']['coverage']['invalid'] == 1
    else:
        assert refreshed['server_id'] == 'B2'
        assert refreshed['quick_log'] == original['quick_log']
        assert reopened['statistics']['coverage']['invalid'] == 0
        assert reopened['statistics']['coverage']['rpr_complete'] == 2
        assert [e for e in refreshed['events'] if e['kind'] != 'serve'] == [
            {**old, 'event_id': new['event_id']} for old, new in zip(
                [e for e in original['events'] if e['kind'] != 'serve'],
                [e for e in refreshed['events'] if e['kind'] != 'serve'])]
    # Reopening is stable; it does not regenerate already-correct event IDs.
    again = restarted._normalize(restarted.get_state(project['project_id']))
    assert again['rallies'] == restarted.get_state(project['project_id'])['rallies']


def test_incompatible_quick_receiver_selection_is_flagged_without_rewriting():
    from models.quick_stats import reconcile_quick_stats
    old = dict(server_id='A2', receiver_id='B2', winner='A', provisional=False)
    category = dict(version=1, kind='error', player_id='B2')
    point = quick_point_stats(log('B2', error={'player_id': 'B2', 'kind': 'receive'}), category, old)
    corrected = {**old, 'receiver_id': 'B1'}
    assert reconcile_quick_stats(point, category, corrected) == point
