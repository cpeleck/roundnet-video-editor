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
