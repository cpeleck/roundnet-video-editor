import React from 'react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Export } from './Export';
import { request } from './api';
import * as api from './api';
import type { Project } from './types';

vi.mock('./Video', () => ({ Video: ({ videoRef, onTime, onPlay, onPause }: any) => <video ref={videoRef} onTimeUpdate={onTime} onPlay={onPlay} onPause={onPause}/> }));
vi.mock('./api', async importOriginal => ({ ...await importOriginal<typeof import('./api')>(), request: vi.fn() }));
beforeEach(() => {
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
});
afterEach(() => { cleanup(); vi.clearAllMocks(); vi.restoreAllMocks(); });
const project = (settings: Record<string, unknown> = {}): Project => ({
  project_id: 'p', revision: 1, video_duration: 10,
  video_metadata: { width: 1920, height: 1080, fps: 30 },
  rallies: [{ rally_id: 'one', start_time: 0, end_time: 1, enabled: true,
    rejected: false, classification: {}, crop_keyframes: [], note: '' }],
  timeline: [], match_settings: { team_a: 'A', team_b: 'B' },
  statistics: { coverage: { unresolved: 0 } }, highlight_queue: [],
  export_settings: { output_path: '/tmp/output.mp4', ...settings },
} as unknown as Project);

test('phone export needs no Mac destination and offers a completed video download', async () => {
  vi.spyOn(api, 'isPhoneConnection').mockReturnValue(true);
  const command = vi.fn().mockResolvedValue({ ...project(), revision: 2 });
  vi.mocked(request).mockResolvedValue({ id: 'phone-render', kind: 'export' });
  const props = { project: project({ output_path: '' }), command, onJob: vi.fn(), onError: vi.fn(), busy: false };
  const { rerender } = render(<Export {...props} job={null}/>);
  expect(screen.queryByText('Save video to')).toBeNull();
  expect((screen.getByRole('button', { name: 'Export video' }) as HTMLButtonElement).disabled).toBe(false);
  fireEvent.click(screen.getByRole('button', { name: 'Export video' }));
  await waitFor(() => expect(request).toHaveBeenCalledWith('/projects/p/export', expect.objectContaining({ expected_revision: 2 })));
  rerender(<Export {...props} job={{ id: 'phone-render', kind: 'export', status: 'succeeded', progress: 100, message: 'Done' }}/>);
  expect(screen.getByRole('link', { name: 'Download MP4' }).getAttribute('href')).toBe('/api/jobs/phone-render/output');
});

test('quality selectors save preferences and submit 4K at 60 fps', async () => {
  const command = vi.fn().mockResolvedValue({ ...project(), revision: 2 });
  vi.mocked(request).mockResolvedValue({ id: 'render', kind: 'export' });
  const onJob = vi.fn();
  render(<Export project={project()} command={command} job={null} onJob={onJob} onError={vi.fn()} busy={false}/>);
  fireEvent.change(screen.getByLabelText('Resolution'), { target: { value: '2160p' } });
  fireEvent.change(screen.getByLabelText('Frame rate'), { target: { value: '60' } });
  expect(command).toHaveBeenLastCalledWith('export_settings', { settings: expect.objectContaining({ resolution: '2160p', frame_rate: 60 }) });
  fireEvent.click(screen.getByRole('button', { name: 'Export video' }));
  await waitFor(() => expect(request).toHaveBeenCalledWith('/projects/p/export', expect.objectContaining({
    options: expect.objectContaining({ resolution: '2160p', frame_rate: 60 }), expected_revision: 2,
  })));
  expect(onJob).toHaveBeenCalledWith(expect.objectContaining({ id: 'render' }));
});

test('saved quality settings restore and size the end-card preview', () => {
  render(<Export project={project({ resolution: '2160p', frame_rate: 60 })} command={vi.fn()} job={null} onJob={vi.fn()} onError={vi.fn()} busy={false}/>);
  expect((screen.getByLabelText('Resolution') as HTMLSelectElement).value).toBe('2160p');
  expect((screen.getByLabelText('Frame rate') as HTMLSelectElement).value).toBe('60');
  fireEvent.click(screen.getByRole('button', { name: 'End card' }));
  expect(screen.getByAltText('Rendered end card matching the final video').getAttribute('src')).toContain('resolution=2160p');
});

test('preview advances team scores at clip boundaries and omits serving details', () => {
  const initial = project();
  initial.rallies = [0, 1, 2].map(i => ({ ...initial.rallies[0], rally_id: String(i), start_time: i, end_time: i + 1 }));
  initial.match_settings.players = ['A1', 'A2', 'B1', 'B2'].map(player_id => ({ player_id, name: player_id, team: player_id[0] }));
  initial.timeline = [
    { server_id: 'A1', receiver_id: 'B1', score_before: [0, 0], score_after: [1, 0], next_server_id: 'B2', next_receiver_id: 'A2' },
    { server_id: 'B2', receiver_id: 'A2', score_before: [1, 0], score_after: [1, 1], next_server_id: 'B2', next_receiver_id: 'A1' },
    { server_id: 'B2', receiver_id: 'A1', score_before: [1, 1], score_after: [1, 2], next_server_id: 'A2', next_receiver_id: 'B1' },
  ].map((row, i) => ({ ...row, index: i, rally_id: String(i), provisional: false, provisional_after: false, winner: i ? 'B' : 'A', kind: 'ace' }));
  render(<Export project={initial} command={vi.fn()} job={null} onJob={vi.fn()} onError={vi.fn()} busy={false}/>);
  const video = document.querySelector('video')!;
  const card = () => screen.getByRole('status', { name: 'Video scorecard' }).textContent;
  expect(card()).toContain('A 0 | B 0');
  expect(card()).not.toContain('Server:');
  expect(card()).not.toContain('Receiver:');
  fireEvent.play(video);
  video.currentTime = 1; fireEvent.timeUpdate(video);
  expect(card()).toContain('A 1 | B 0');
  video.currentTime = 2; fireEvent.timeUpdate(video);
  expect(card()).toContain('A 1 | B 1');
  video.currentTime = 3; fireEvent.timeUpdate(video);
  expect(card()).toContain('A 1 | B 2');
  expect(card()).not.toContain('Server:');
  expect(card()).not.toContain('Receiver:');
});

test('highlight reel trim accepts MM:SS without changing match clip bounds', async () => {
  const initial = project({ mode: 'highlights' }); initial.video_duration = 180;
  initial.rallies[0].start_time = 65; initial.rallies[0].end_time = 75;
  initial.highlight_queue = [{ rally_id: 'one' }];
  const command = vi.fn().mockResolvedValue(initial);
  render(<Export project={initial} command={command} job={null} onJob={vi.fn()} onError={vi.fn()} busy={false}/>);
  const start = screen.getByLabelText('In (MM:SS)') as HTMLInputElement;
  const end = screen.getByLabelText('Out (MM:SS)') as HTMLInputElement;
  expect(start.value).toBe('01:05'); expect(end.value).toBe('01:15');
  fireEvent.change(start, { target: { value: '01:06.5' } });
  fireEvent.change(end, { target: { value: '01:14' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save reel trim' }));
  await waitFor(() => expect(command).toHaveBeenCalledWith('highlight_queue', { clips: [{ rally_id: 'one', start_time: 66.5, end_time: 74 }] }));
  expect(initial.rallies[0].start_time).toBe(65);
  expect(initial.rallies[0].end_time).toBe(75);
});
