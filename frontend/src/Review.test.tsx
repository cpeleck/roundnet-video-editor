import React, { useState } from 'react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Review } from './Review';
import type { Project } from './types';
vi.mock('./Video', () => ({ Video: ({ videoRef, onTime, onPlay, onPause, onLoaded, onEnded }: any) => <video ref={videoRef} onTimeUpdate={onTime} onPlay={onPlay} onPause={onPause} onLoadedMetadata={onLoaded} onEnded={onEnded}/> }));
const rally = (id: string, start: number, end: number) => ({ rally_id: id, start_time: start, end_time: end, enabled: true, rejected: false, starred: false, reviewed: false, confidence: 0, classification: {}, outcome: '', winner: '', note: '', crop_keyframes: [], point_stats: {} });
const project = (): Project => ({ project_id: 'p', revision: 1, position: 0, video_duration: 60, selected: 0, rallies: [rally('one', 2, 8), rally('two', 20, 25)], timeline: [0, 1].map(i => ({ index: i, rally_id: i ? 'two' : 'one', server_id: 'A1', receiver_id: 'B1', score_before: [0, 0], provisional: false, winner: '', kind: '' })), match_settings: { team_a: 'North', team_b: 'South', players: ['A1','A2','B1','B2'].map(player_id => ({ player_id, team: player_id[0], name: player_id })), starting_server: 'A1', starting_receiver: 'B1', initial_score_a: 0, initial_score_b: 0, target_score: 21 }, statistics: { players: [], teams: {}, coverage: {}, points: [], warnings: [], rpr_eligible: false }, video_metadata: { width: 640, height: 360, fps: 30 }, settings: {}, highlight_queue: [], export_settings: {} } as unknown as Project);
let paused = true;
let play: ReturnType<typeof vi.fn>;
let pause: ReturnType<typeof vi.fn>;
beforeEach(() => {
  paused = true;
  play = vi.fn(function(this: HTMLVideoElement) { paused = false; this.dispatchEvent(new Event('play')); return Promise.resolve(); });
  pause = vi.fn(function(this: HTMLVideoElement) { paused = true; this.dispatchEvent(new Event('pause')); });
  Object.defineProperty(HTMLMediaElement.prototype, 'play', { configurable: true, value: play });
  Object.defineProperty(HTMLMediaElement.prototype, 'pause', { configurable: true, value: pause });
  Object.defineProperty(HTMLMediaElement.prototype, 'paused', { configurable: true, get: () => paused });
});
afterEach(cleanup);
const video = () => document.querySelector('video')!;
const button = (name: string | RegExp) => screen.getByRole('button', { name });
function mount(command = vi.fn().mockResolvedValue(project()), initial = project()) {
  const result = render(<Review project={initial} command={command} busy={false} shortcuts onStats={() => {}}/>);
  return { ...result, command };
}

test('markers capture actual video time, retain failures, and reject reversed ends', async () => {
  const command = vi.fn().mockResolvedValue(undefined);
  mount(command);
  video().currentTime = 10.123;
  fireEvent.click(button('Mark start'));
  video().currentTime = 9;
  fireEvent.click(button('Mark end'));
  expect(screen.getByRole('alert').textContent).toContain('after its start');
  expect(command).not.toHaveBeenCalled();
  video().currentTime = 15.456;
  fireEvent.keyDown(window, { key: 'x' });
  await waitFor(() => expect(command).toHaveBeenCalledWith('add', { start_time: 10.123, end_time: 15.456 }));
  expect(button('Mark end')).toBeTruthy();
  command.mockResolvedValue(project());
  fireEvent.click(button('Mark end'));
  await waitFor(() => expect(button('Mark start')).toBeTruthy());
  expect(video().currentTime).toBe(15.456);
  expect(paused).toBe(true);
});

test('classification resumes at its own end despite next selection and continues past it', async () => {
  const command = vi.fn(async (..._args: any[]) => ({ ...project(), selected: 1, position: 8 }));
  function Harness() {
    const [p, setP] = useState(project());
    return <Review project={p} busy={false} shortcuts onStats={() => {}} command={async (...args) => { const next = await command(...args); setP(next); return next; }}/>;
  }
  render(<Harness/>);
  fireEvent.click(button(/Ace.*Unreturnable/));
  await waitFor(() => expect(play).toHaveBeenCalled());
  expect(video().currentTime).toBe(8);
  expect(paused).toBe(false);
  video().currentTime = 26;
  fireEvent.timeUpdate(video());
  expect(paused).toBe(false);
  fireEvent.ended(video());
  expect(button('Play video')).toBeTruthy();
});

test('finishing the final clip shows its updated score and next serving pair', async () => {
  const initial = project(); initial.rallies = [initial.rallies[0]]; initial.timeline = [initial.timeline[0]];
  const row = { ...initial.timeline[0], score_after: [1, 0], next_server_id: 'B2', next_receiver_id: 'A2', provisional_after: false };
  function Harness() {
    const [p, setP] = useState(initial);
    return <Review project={p} busy={false} shortcuts onStats={() => {}} command={async () => {
      const next = { ...initial, position: 8, timeline: [{ ...row, winner: 'A', kind: 'ace' }] };
      setP(next); return next;
    }}/>;
  }
  render(<Harness/>);
  fireEvent.click(button(/Ace.*Unreturnable/));
  await waitFor(() => expect(screen.getByText('SCORE AFTER THIS CLIP')).toBeTruthy());
  expect(document.querySelector('.scoreboard strong')?.textContent).toBe('1–0');
  expect(document.querySelector('.serve-assignment')?.textContent).toContain('Server A1');
  expect(document.querySelector('.serve-assignment')?.textContent).toContain('Receiver B1');
  expect(document.querySelector('.next-serve-assignment')?.textContent).toContain('Next server B2');
  expect(document.querySelector('.next-serve-assignment')?.textContent).toContain('Next receiver A2');
  video().currentTime = 3; fireEvent.timeUpdate(video());
  expect(screen.getByText('SCORE ENTERING THIS CLIP')).toBeTruthy();
  expect(document.querySelector('.scoreboard strong')?.textContent).toBe('0–0');
});

test('five-second controls use actual time and clamp; inputs ignore shortcuts', () => {
  mount();
  video().currentTime = 12;
  fireEvent.keyDown(window, { key: 'ArrowRight' });
  expect(video().currentTime).toBe(17);
  fireEvent.click(button('Back five seconds'));
  expect(video().currentTime).toBe(12);
  video().currentTime = 59;
  fireEvent.click(button('Forward five seconds'));
  expect(video().currentTime).toBe(60);
  video().currentTime = 1;
  fireEvent.keyDown(window, { key: 'ArrowLeft' });
  expect(video().currentTime).toBe(0);
  const input = screen.getByRole('slider');
  fireEvent.keyDown(input, { key: 'ArrowRight' });
  expect(video().currentTime).toBe(0);
});

test('clip replay stops at its end and autoplay failures expose Resume', async () => {
  mount();
  fireEvent.click(button('Next'));
  await act(async () => {});
  fireEvent.click(button('Play video'));
  await waitFor(() => expect(play).toHaveBeenCalled());
  video().currentTime = 26;
  fireEvent.timeUpdate(video());
  expect(paused).toBe(true);
  play.mockRejectedValueOnce(new Error('autoplay denied'));
  fireEvent.click(button(/Ace.*Unreturnable/));
  await waitFor(() => expect(button('Resume')).toBeTruthy());
});

test('guided details pause; cancel and failed save leave previous outcome intact', async () => {
  const { command } = mount(vi.fn().mockResolvedValue(undefined));
  fireEvent.click(button(/Defensive Hold.*Serving/));
  expect(paused).toBe(true);
  expect(command).not.toHaveBeenCalled();
  fireEvent.click(button('Add possession'));
  fireEvent.change(screen.getByLabelText('First touch 2'), { target: { value: 'A2' } });
  fireEvent.change(screen.getByLabelText('Hitter 2'), { target: { value: 'A1' } });
  fireEvent.change(screen.getByLabelText('Final defender'), { target: { value: 'B2' } });
  fireEvent.click(button('Save point & continue'));
  await waitFor(() => expect(command).toHaveBeenCalledWith('classify', expect.objectContaining({ quick_log: expect.objectContaining({ terminal_defender_id: 'B2', possessions: [expect.any(Object), expect.objectContaining({ first_touch_id: 'A2', hitter_id: 'A1' })] }) })));
  expect(screen.getByLabelText('Hitter 2')).toBeTruthy();
  expect(play).not.toHaveBeenCalled();
  fireEvent.click(button('Cancel'));
  expect(screen.queryByLabelText('Guided possession log')).toBeNull();
});

test('keyboard outcome focus cannot score the next point through Space', async () => {
  const { command } = mount();
  fireEvent.keyDown(window, { key: '1' });
  await waitFor(() => expect(play).toHaveBeenCalled());
  expect(document.activeElement?.getAttribute('aria-label')).toBe('Selected clip outcome');
  fireEvent.keyDown(document.activeElement!, { key: ' ' });
  expect(command.mock.calls.filter(c => c[0] === 'classify')).toHaveLength(1);
});

function liveMount(initial = project(), reduce = (type: string, values: Record<string, unknown> | undefined, p: Project): Project | undefined | Promise<Project | undefined> => {
  if (type === 'select') {
    const selected = p.rallies.findIndex(r => r.rally_id === values?.rally_id);
    return { ...p, selected, position: p.rallies[selected].start_time };
  }
  return p;
}) {
  const command = vi.fn(reduce);
  function Harness() {
    const [p, setP] = useState(initial);
    return <Review project={p} busy={false} shortcuts onStats={() => {}} command={async (type, values) => {
      const next = await command(type, values, p);
      if (next) setP(next);
      return next;
    }}/>;
  }
  return { ...render(<Harness/>), command };
}

test.each(['keyboard', 'button'])('resetting a pending start using %s replaces the marker and clears errors', async method => {
  const { command } = mount();
  video().currentTime = 10;
  fireEvent.keyDown(window, { key: 'w' });
  video().currentTime = 9;
  fireEvent.keyDown(window, { key: 'x' });
  expect(screen.getByRole('alert').textContent).toContain('after its start');
  video().currentTime = 12.345;
  if (method === 'keyboard') fireEvent.keyDown(window, { key: 'w' });
  else fireEvent.click(button('Reset start to playhead'));
  expect(screen.queryByRole('alert')).toBeNull();
  expect(document.querySelector('.notice')?.textContent).toContain('00:12.3');
  video().currentTime = 15.678;
  fireEvent.keyDown(window, { key: 'x' });
  await waitFor(() => expect(command).toHaveBeenCalledWith('add', { start_time: 12.345, end_time: 15.678 }));
});

test('Escape cancels a reset marker and pending markers block clip actions', () => {
  const { command } = mount();
  video().currentTime = 10;
  fireEvent.keyDown(window, { key: 'w' });
  expect((button('Delete clip') as HTMLButtonElement).disabled).toBe(true);
  expect((button('Continue after clip') as HTMLButtonElement).disabled).toBe(true);
  fireEvent.keyDown(window, { key: ']' });
  fireEvent.keyDown(window, { key: '1' });
  expect(command).not.toHaveBeenCalled();
  video().currentTime = 12;
  fireEvent.keyDown(window, { key: 'w' });
  fireEvent.keyDown(window, { key: 'Escape' });
  fireEvent.keyDown(window, { key: 'x' });
  expect(command).not.toHaveBeenCalled();
  expect(button('Mark start')).toBeTruthy();
});

test.each(['paused', 'playing', 'looping'])('continue after a %s clip exits its range and subsequent Play stays in source mode', async mode => {
  liveMount();
  fireEvent.click(button('Next'));
  await waitFor(() => expect(video().currentTime).toBe(20));
  if (mode !== 'paused') fireEvent.click(button('Play video'));
  if (mode === 'looping') {
    fireEvent.click(screen.getByLabelText('Loop'));
    video().currentTime = 26; fireEvent.timeUpdate(video());
    expect(video().currentTime).toBe(20);
  }
  fireEvent.click(button('Continue after clip'));
  await waitFor(() => expect(paused).toBe(false));
  expect(video().currentTime).toBe(25);
  expect((screen.getByLabelText('Loop') as HTMLInputElement).checked).toBe(false);
  video().currentTime = 40; fireEvent.timeUpdate(video());
  expect(paused).toBe(false);
  fireEvent.click(button('Pause video'));
  fireEvent.click(button('Play video'));
  expect(video().currentTime).toBe(40);
  expect(paused).toBe(false);
  // Selecting a clip again deliberately restores bounded playback.
  fireEvent.click(button('Previous'));
  await waitFor(() => expect(video().currentTime).toBe(2));
  fireEvent.click(button('Play video'));
  video().currentTime = 9; fireEvent.timeUpdate(video());
  expect(paused).toBe(true);
});

test('continue handles autoplay rejection and the end of the recording', async () => {
  const initial = project(); initial.rallies[0].end_time = 60;
  const { rerender } = mount(undefined, initial);
  fireEvent.click(button('Play video'));
  expect(paused).toBe(false);
  play.mockClear();
  fireEvent.click(button('Continue after clip'));
  expect(video().currentTime).toBe(60);
  expect(paused).toBe(true);
  expect(play).not.toHaveBeenCalled();
  rerender(<Review project={project()} command={vi.fn()} busy={false} shortcuts onStats={() => {}}/>);
  play.mockRejectedValueOnce(new Error('autoplay denied'));
  fireEvent.click(button('Continue after clip'));
  await waitFor(() => expect(button('Resume')).toBeTruthy());
  expect(video().currentTime).toBe(8);
  fireEvent.click(button('Resume'));
  await waitFor(() => expect(paused).toBe(false));
  video().currentTime = 30; fireEvent.timeUpdate(video());
  expect(paused).toBe(false);
});

test.each([false, true])('deleting a clip pauses in source mode with only clip = %s', async onlyClip => {
  const initial = project(); if (onlyClip) initial.rallies = [initial.rallies[0]];
  const { command } = liveMount(initial, (type, values, p) => {
    if (type !== 'delete') return p;
    const removed = p.rallies.find(r => r.rally_id === values?.rally_id)!;
    return { ...p, rallies: p.rallies.filter(r => r !== removed), selected: onlyClip ? -1 : 0, position: removed.end_time,
      timeline: p.timeline.filter(t => t.rally_id !== removed.rally_id), explanation: 'Clip deleted. Undo restores it.', can_undo: true };
  });
  fireEvent.click(button('Play video'));
  fireEvent.click(button('Delete clip'));
  await waitFor(() => expect(document.querySelectorAll('.filmstrip-clip')).toHaveLength(onlyClip ? 0 : 1));
  expect(command).toHaveBeenCalledWith('delete', { rally_id: 'one' }, initial);
  expect(video().currentTime).toBe(8);
  expect(paused).toBe(true);
  expect(button('Undo')).toBeTruthy();
  expect(screen.queryByText('Not a rally')).toBeNull();
  fireEvent.click(button('Play video'));
  expect(video().currentTime).toBe(8);
  video().currentTime = 30; fireEvent.timeUpdate(video());
  expect(paused).toBe(false);
});

test('failed deletion keeps selection and playback; pending deletion blocks repeated edits', async () => {
  let resolve!: (value: Project | undefined) => void;
  const command = vi.fn(() => new Promise<Project | undefined>(r => { resolve = r; }));
  mount(command);
  video().currentTime = 5;
  fireEvent.click(button('Play video'));
  fireEvent.click(button('Delete clip'));
  expect((button('Deleting…') as HTMLButtonElement).disabled).toBe(true);
  expect((button('Next') as HTMLButtonElement).disabled).toBe(true);
  fireEvent.keyDown(window, { key: 'w' });
  fireEvent.keyDown(window, { key: '1' });
  expect(command).toHaveBeenCalledTimes(1);
  await act(async () => resolve(undefined));
  expect(button('Delete clip')).toBeTruthy();
  expect(document.querySelectorAll('.filmstrip-clip')).toHaveLength(2);
  expect(video().currentTime).toBe(5);
  expect(paused).toBe(false);
  expect(document.querySelector('.filmstrip-clip.selected')?.textContent).toContain('00:02.0');
});

test.each([false, true])('outdated player logs have a direct correction action with quick log = %s', quick => {
  const initial = project();
  initial.rallies[0].classification = { kind: 'defensive_hold' };
  initial.rallies[0].point_stats = { version: 1, complete: true, server_id: 'A1', receiver_id: 'B2',
    events: [{ event_id: 'serve', player_id: 'A1', kind: 'serve', result: 'in' }],
    ...(quick ? { quick_log: { version: 1 as const, possessions: [{ first_touch_id: 'B2', hitter_id: 'B2' }, { first_touch_id: 'A1', hitter_id: 'A1' }], serve_faults: [], error: null, terminal_defender_id: null } } : {}) };
  mount(undefined, initial);
  expect(screen.getByRole('alert').textContent).toContain('Player log needs review');
  fireEvent.click(button('Review player log'));
  if (quick) {
    expect(screen.getByLabelText('Guided possession log')).toBeTruthy();
    expect((screen.getByLabelText('First touch 1') as HTMLSelectElement).disabled).toBe(false);
  } else expect(screen.getByRole('dialog').textContent).toContain('Who touched it next?');
  expect((button('Delete clip') as HTMLButtonElement).disabled).toBe(true);
  expect((button('Continue after clip') as HTMLButtonElement).disabled).toBe(true);
});

test('assignments stay attached to clip identity after chronological reordering', () => {
  const initial = project();
  const { rerender } = mount(undefined, initial);
  const reordered = { ...initial, selected: 1, rallies: [initial.rallies[1], { ...initial.rallies[0], start_time: 30, end_time: 35 }],
    timeline: [{ ...initial.timeline[1], index: 0, server_id: 'A1', receiver_id: 'B1' },
      { ...initial.timeline[0], index: 1, server_id: 'B2', receiver_id: 'A2' }] };
  rerender(<Review project={reordered} command={vi.fn()} busy={false} shortcuts onStats={() => {}}/>);
  expect(document.querySelector('.filmstrip-clip.selected')?.textContent).toContain('00:30.0');
  expect(document.querySelector('.serve-assignment')?.textContent).toContain('Server B2');
  expect(document.querySelector('.serve-assignment')?.textContent).toContain('Receiver A2');
});

test('a draft reports its editing lock and releases it on cancel and unmount', () => {
  const onDraftChange = vi.fn();
  const { unmount } = render(<Review project={project()} command={vi.fn()} busy={false} shortcuts onStats={() => {}} onDraftChange={onDraftChange}/>);
  fireEvent.keyDown(window, { key: 'w' });
  expect(onDraftChange).toHaveBeenLastCalledWith(true);
  fireEvent.keyDown(window, { key: 'Escape' });
  expect(onDraftChange).toHaveBeenLastCalledWith(false);
  fireEvent.click(button(/Defensive Hold.*Serving/));
  expect(onDraftChange).toHaveBeenLastCalledWith(true);
  unmount();
  expect(onDraftChange).toHaveBeenLastCalledWith(false);
});

test('clip retrimming uses MM:SS and saves source seconds', async () => {
  const initial = project(); initial.video_duration = 300;
  initial.rallies[0] = { ...initial.rallies[0], start_time: 83.125, end_time: 95.5 };
  const { command } = mount(undefined, initial);
  const start = screen.getByLabelText('In (MM:SS)') as HTMLInputElement;
  const end = screen.getByLabelText('Out (MM:SS)') as HTMLInputElement;
  expect(start.value).toBe('01:23.125');
  expect(end.value).toBe('01:35.5');
  fireEvent.change(start, { target: { value: '01:24.25' } });
  fireEvent.change(end, { target: { value: '01:36' } });
  fireEvent.click(button('Save trim'));
  await waitFor(() => expect(command).toHaveBeenCalledWith('trim', { rally_id: 'one', start_time: 84.25, end_time: 96 }));
});

test('trim validates formatted times and preserves untouched precision when using the playhead', async () => {
  const initial = project(); initial.video_duration = 120;
  initial.rallies[0].start_time = 2.123456789;
  const { command } = mount(undefined, initial);
  const end = screen.getByLabelText('Out (MM:SS)');
  for (const value of ['8', '00:60', '00:01', '02:01']) {
    fireEvent.change(end, { target: { value } });
    expect((button('Save trim') as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(button('Save trim'));
  }
  expect(command).not.toHaveBeenCalled();
  video().currentTime = 65.25; fireEvent.timeUpdate(video());
  fireEvent.click(screen.getAllByRole('button', { name: 'Use playhead' })[1]);
  expect((end as HTMLInputElement).value).toBe('01:05.25');
  fireEvent.click(button('Save trim'));
  await waitFor(() => expect(command).toHaveBeenCalledWith('trim', { rally_id: 'one', start_time: 2.123456789, end_time: 65.25 }));
});
