import { afterEach, expect, test, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Benchmark } from './Benchmark';
import { request } from './api';
import type { Project, ProjectSummary } from './types';

vi.mock('./api', () => ({ request: vi.fn() }));
afterEach(() => { cleanup(); vi.resetAllMocks(); });
const project = { project_id: 'p1', title: 'Match one', revision: 7 } as Project;
const projects = [project] as ProjectSummary[];
const metric = { precision: .9, recall: .8, median_boundary_error_seconds: .25, true_positives: 9, false_positives: 1, false_negatives: 2 };
const report = { report_id: 'report1', created_utc: '2026-11-05T15:00:00Z', recording_count: 3,
  learned_events: metric, heuristic_events: metric, motion_events: { ...metric, precision: .7 },
  warnings: ['Collect 8–10 varied recordings.'], boundary_semantics: 'Errors use reviewed source boundaries.',
  runtime_definition: 'Measured detector analysis time.', correction_definition: 'Net interval changes.',
  folds: [{ held_out_source_id: 'source1', source_title: 'Match one', duration_seconds: 20,
    learned_events: metric, heuristic_events: metric, motion_events: metric,
    reference_intervals: [{ start: 2, end: 6 }], predicted_intervals: { learned: [{ start: 2.1, end: 6.1 }], motion: [{ start: 1, end: 6 }] } }],
  operations: { timed_recording_count: 0, measured_25_minute_count: 0, median_measured_25_minute_seconds: null,
    estimated_25_minute_seconds: null, median_corrections_per_recording: 2, correction_recording_count: 3,
    recordings: [{ source_id: 'source1', title: 'Match one', duration_seconds: 20, detection_seconds: null,
      corrections: { added: 1, removed: 1, trimmed: 0, total: 2 } }] } };
const recording = (id: string) => ({ id, source_id: id, project_id: 'p1', title: `Match ${id}`, revision: 7,
  duration_seconds: 20, rally_count: 2, saved_utc: '2026-11-05T15:00:00Z', detection_run: null,
  corrections: { added: 0, removed: 1, trimmed: 0, total: 1 } });
const overview = (count = 0) => ({ recordings: Array.from({ length: count }, (_, i) => recording(String(i))), reports: [], active_job: null });
function mount() { return render(<Benchmark project={project} projects={projects} onOpen={vi.fn()} onBack={vi.fn()}/>); }

test('ground truth requires confirmation and captures the latest project revision', async () => {
  vi.mocked(request).mockImplementation(async path => {
    if (path === '/projects/p1') return { ...project, revision: 8 };
    return overview();
  });
  mount();
  const save = screen.getByRole('button', { name: 'Save ground truth snapshot' });
  await screen.findByText('No labeled recordings saved yet. Start with one corrected match.');
  expect((save as HTMLButtonElement).disabled).toBe(true);
  fireEvent.click(screen.getByRole('checkbox', { name: /I checked the whole/ }));
  fireEvent.click(save);
  await waitFor(() => expect(request).toHaveBeenCalledWith('/admin/benchmarks/recordings', {
    project_id: 'p1', expected_revision: 8, full_review_confirmed: true,
  }));
  await waitFor(() => expect((screen.getByRole('checkbox', { name: /I checked the whole/ }) as HTMLInputElement).checked).toBe(false));
});

test('evaluation requires three selected recordings and submits only selected snapshots', async () => {
  vi.mocked(request).mockImplementation(async path => path.endsWith('/run') ? { id: 'job1', status: 'queued', message: 'Waiting' } : overview(3));
  mount();
  const evaluate = await screen.findByRole('button', { name: 'Evaluate 3 recordings' });
  expect((evaluate as HTMLButtonElement).disabled).toBe(false);
  fireEvent.click(screen.getByRole('checkbox', { name: 'Include Match 1' }));
  expect((screen.getByRole('button', { name: 'Evaluate 2 recordings' }) as HTMLButtonElement).disabled).toBe(true);
  fireEvent.click(screen.getByRole('checkbox', { name: 'Include Match 1' }));
  fireEvent.click(screen.getByRole('button', { name: 'Evaluate 3 recordings' }));
  await waitFor(() => expect(request).toHaveBeenCalledWith('/admin/benchmarks/run', { recording_ids: ['0', '2', '1'] }));
  expect(await screen.findByRole('button', { name: 'Cancel evaluation' })).toBeTruthy();
});

test('saved report compares detectors, shows missing timing, and exposes exact source intervals', async () => {
  vi.mocked(request).mockImplementation(async path => path.includes('/reports/') ? report : {
    ...overview(3), reports: [{ id: 'report1', recording_count: 3, created_utc: report.created_utc }],
  });
  mount();
  expect(await screen.findByRole('heading', { name: '3. Side-by-side results' })).toBeTruthy();
  const table = screen.getByRole('table', { name: /Rally metrics pooled/ });
  expect(table.textContent).toContain('0.90');
  expect(table.textContent).toContain('0.70');
  expect(table.textContent).toContain('0.25 s');
  expect(screen.getAllByText('Unavailable').length).toBeGreaterThanOrEqual(2);
  expect(screen.getByRole('link', { name: 'Download report JSON' }).getAttribute('href')).toBe('/api/admin/benchmarks/reports/report1/download');
  expect(screen.getByLabelText('Rally intervals across the source recording')).toBeTruthy();
  fireEvent.click(screen.getByText('Exact source-time intervals'));
  expect(screen.getByText('2.000 – 6.000 s')).toBeTruthy();
});

test('failed capture preserves the confirmation and shows the server error', async () => {
  vi.mocked(request).mockImplementation(async path => {
    if (path === '/projects/p1') return project;
    if (path.endsWith('/recordings')) throw new Error('Run detection before saving ground truth.');
    return overview();
  });
  mount();
  await screen.findByText('No labeled recordings saved yet. Start with one corrected match.');
  fireEvent.click(screen.getByRole('checkbox', { name: /I checked the whole/ }));
  fireEvent.click(screen.getByRole('button', { name: 'Save ground truth snapshot' }));
  expect((await screen.findByRole('alert')).textContent).toContain('Run detection');
  expect((screen.getByRole('checkbox', { name: /I checked the whole/ }) as HTMLInputElement).checked).toBe(true);
});

test('completed background job loads the saved report', async () => {
  vi.mocked(request).mockImplementation(async path => {
    if (path === '/jobs/job1') return { id: 'job1', status: 'succeeded', result: report, message: 'Benchmark complete' };
    if (path.endsWith('/run')) return { id: 'job1', status: 'queued', message: 'Waiting for a worker' };
    return overview(3);
  });
  mount();
  fireEvent.click(await screen.findByRole('button', { name: 'Evaluate 3 recordings' }));
  expect(await screen.findByRole('heading', { name: '3. Side-by-side results' }, { timeout: 2500 })).toBeTruthy();
});
