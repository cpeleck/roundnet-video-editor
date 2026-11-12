import { useEffect, useState } from 'react';
import { request } from './api';
import type { Job, Project, ProjectSummary } from './types';
import { formatTimecode as timecode } from './timecode';

type Interval = { start: number; end: number };
type Metrics = { precision: number | null; recall: number | null; median_boundary_error_seconds: number | null; true_positives: number; false_positives: number; false_negatives: number; recording_count?: number };
type Counts = { added: number; removed: number; trimmed: number; total: number };
type Recording = { id: string; source_id: string; project_id: string; title: string; revision: number; duration_seconds: number; rally_count: number; saved_utc: string; corrections: Counts | null; detection_run: { detection_elapsed_seconds?: number } | null };
type Fold = { held_out_source_id: string; source_title: string; duration_seconds: number; learned_events?: Metrics; heuristic_events?: Metrics; motion_events?: Metrics; reference_intervals?: Interval[]; predicted_intervals?: Record<string, Interval[]> };
type Report = { report_id: string; created_utc: string; recording_count: number; learned_events: Metrics | null; heuristic_events: Metrics | null; motion_events: Metrics | null; folds: Fold[]; warnings: string[]; boundary_semantics: string; correction_definition: string; runtime_definition: string; operations: { timed_recording_count: number; measured_25_minute_count: number; median_measured_25_minute_seconds: number | null; estimated_25_minute_seconds: number | null; median_corrections_per_recording: number | null; correction_recording_count: number; recordings: { source_id: string; title: string; duration_seconds: number; detection_seconds: number | null; corrections: Counts | null }[] } };
type Overview = { recordings: Recording[]; reports: { id: string; created_utc: string; recording_count: number }[]; active_job: Job | null };
const methods = [['learned', 'Learned · held out'], ['heuristic', 'Existing heuristic'], ['motion', 'Motion only']] as const;
const decimal = (value: number | null | undefined, places = 2) => value == null ? '—' : value.toFixed(places);
const seconds = (value: number | null | undefined) => value == null ? 'Unavailable' : `${value.toFixed(2)} s`;

function MetricCells({ metrics }: { metrics?: Metrics | null }) {
  return <><td>{decimal(metrics?.precision)}</td><td>{decimal(metrics?.recall)}</td><td>{metrics?.median_boundary_error_seconds == null ? '—' : seconds(metrics.median_boundary_error_seconds)}</td><td>{metrics ? `${metrics.true_positives} / ${metrics.false_positives} / ${metrics.false_negatives}` : '—'}</td></>;
}

export function Benchmark({ projects, project, onOpen, onBack }: { projects: ProjectSummary[]; project: Project | null; onOpen: (id: string) => void; onBack: () => void }) {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [captureId, setCaptureId] = useState(project?.project_id ?? projects[0]?.project_id ?? '');
  const [confirmed, setConfirmed] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const [job, setJob] = useState<Job | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [foldId, setFoldId] = useState('');
  const [reportId, setReportId] = useState('');
  const [loading, setLoading] = useState(true);
  const fail = (e: unknown) => setError(e instanceof Error ? e.message : String(e));
  const refresh = async () => {
    const next = await request<Overview>('/admin/benchmarks');
    setOverview(next); setSelected(next.recordings.map(row => row.id));
    return next;
  };
  const loadReport = async (id: string) => {
    const value = await request<Report>(`/admin/benchmarks/reports/${id}`);
    setReport(value); setReportId(id); setFoldId(value.folds[0]?.held_out_source_id ?? '');
  };
  useEffect(() => {
    let disposed = false;
    void request<Overview>('/admin/benchmarks').then(async next => {
      if (disposed) return;
      setOverview(next); setSelected(next.recordings.map(row => row.id)); setJob(next.active_job);
      if (next.reports[0]) {
        const value = await request<Report>(`/admin/benchmarks/reports/${next.reports[0].id}`);
        if (!disposed) { setReport(value); setReportId(next.reports[0].id); setFoldId(value.folds[0]?.held_out_source_id ?? ''); }
      }
    }).catch(e => { if (!disposed) fail(e); }).finally(() => { if (!disposed) setLoading(false); });
    return () => { disposed = true; };
  }, []);
  useEffect(() => {
    if (!job || !['queued', 'running'].includes(job.status)) return;
    let disposed = false;
    const timer = window.setInterval(() => {
      void request<Job>(`/jobs/${job.id}`).then(async next => {
        if (disposed) return;
        setJob(next);
        if (next.status === 'succeeded') {
          const result = next.result as unknown as Report;
          setReport(result); setReportId(result.report_id); setFoldId(result.folds[0]?.held_out_source_id ?? '');
          await refresh();
        } else if (next.status === 'failed') setError(next.error ?? next.message);
      }).catch(e => { if (!disposed) fail(e); });
    }, 1000);
    return () => { disposed = true; clearInterval(timer); };
  }, [job?.id, job?.status]);
  const running = !!job && ['queued', 'running'].includes(job.status);
  const capture = async () => {
    setPending(true); setError('');
    try {
      const value = await request<Project>(`/projects/${captureId}`);
      await request('/admin/benchmarks/recordings', { project_id: captureId, expected_revision: value.revision, full_review_confirmed: confirmed });
      await refresh(); setConfirmed(false);
    } catch (e) { fail(e); } finally { setPending(false); }
  };
  const run = async () => {
    setPending(true); setError('');
    try { setJob(await request<Job>('/admin/benchmarks/run', { recording_ids: selected })); }
    catch (e) { fail(e); } finally { setPending(false); }
  };
  const fold = report?.folds.find(row => row.held_out_source_id === foldId);
  return <main className="benchmark-page">
    <div className="section-heading"><div><div className="eyebrow">LOCAL ADMIN</div><h1>Measure your detector.</h1><p>Use corrected source-time rallies as ground truth across 8–10 varied recordings.</p></div><button className="button" onClick={onBack}>Back to matches</button></div>
    {error && <div className="error-banner" role="alert">{error}</div>}
    <section className="benchmark-panel" aria-labelledby="ground-truth-title">
      <div className="section-heading"><div><h2 id="ground-truth-title">1. Save reviewed ground truth</h2><p>Run detection, trim boundaries, add missed rallies, and mark false detections “Not a rally” in Review.</p></div><span className="count-pill">{overview?.recordings.length ?? 0} / 8–10 recordings</span></div>
      <div className="row benchmark-capture"><label>Reviewed match<select value={captureId} onChange={e => { setCaptureId(e.target.value); setConfirmed(false); }}><option value="">Choose a match</option>{projects.map(p => <option key={p.project_id} value={p.project_id}>{p.title}</option>)}</select></label><button className="button" disabled={!captureId || pending} onClick={() => onOpen(captureId)}>Open in Review</button></div>
      <label className="checkbox"><input type="checkbox" checked={confirmed} onChange={e => setConfirmed(e.target.checked)}/>I checked the whole recording for missed rallies and corrected every true rally’s start and end.</label>
      <p className="helper">All true rallies are included, even those excluded from export. Each save freezes these labels and features; saving the same recording again replaces its dataset entry. Edit the match and save again to update ground truth.</p>
      <button className="button primary" disabled={!confirmed || !captureId || pending || running} onClick={() => void capture()}>{pending ? 'Working…' : 'Save ground truth snapshot'}</button>
      {loading ? <p role="status">Loading benchmark data…</p> : !overview?.recordings.length ? <p className="helper">No labeled recordings saved yet. Start with one corrected match.</p> : <div className="benchmark-table-wrap"><table className="benchmark-table"><caption>Saved ground-truth recordings · one entry per original video</caption><thead><tr><th scope="col">Include / recording</th><th scope="col">Duration</th><th scope="col">True rallies</th><th scope="col">Detection time</th><th scope="col">Corrections</th><th scope="col">Label snapshot</th></tr></thead><tbody>{overview.recordings.map(row => <tr key={row.id}><th scope="row"><label className="checkbox"><input type="checkbox" aria-label={`Include ${row.title}`} checked={selected.includes(row.id)} disabled={running} onChange={e => setSelected(ids => e.target.checked ? [...ids, row.id] : ids.filter(id => id !== row.id))}/>{row.title}</label></th><td>{timecode(row.duration_seconds)}</td><td>{row.rally_count}</td><td>{seconds(row.detection_run?.detection_elapsed_seconds)}</td><td>{row.corrections?.total ?? '—'}</td><td>Revision {row.revision}<small>{new Date(row.saved_utc).toLocaleString()}</small></td></tr>)}</tbody></table></div>}
    </section>
    <section className="benchmark-panel"><h2>2. Run leave-one-recording-out</h2><p>Each fold fits on the other recordings. Both boundaries must be within ±1 second for a one-to-one match. Aim for 8–10 sources; three is the training minimum.</p><p className="helper">The motion baseline uses only normalized motion with fixed 0.55 / 0.38 thresholds and the same saved temporal rules. Reports are saved locally. The fitted profile is saved for inspection.</p><div className="row"><button className="button primary" disabled={loading || pending || running || selected.length < 3} onClick={() => void run()}>Evaluate {selected.length} recordings</button>{running && <button className="button" onClick={() => { void request<Job>(`/jobs/${job!.id}/cancel`, {}).then(setJob).catch(fail); }}>Cancel evaluation</button>}</div>{job && <p role="status" className="helper">{job.message}{job.status === 'succeeded' ? ' · Results saved' : ''}</p>}</section>
    {!!overview?.reports.length && <label className="benchmark-report-select">Saved report<select value={reportId} onChange={e => { setError(''); void loadReport(e.target.value).catch(fail); }}>{overview.reports.map(row => <option key={row.id} value={row.id}>{new Date(row.created_utc).toLocaleString()} · {row.recording_count} recordings</option>)}</select></label>}
    {report && <section className="benchmark-panel" aria-labelledby="comparison-title"><div className="section-heading"><div><h2 id="comparison-title">3. Side-by-side results</h2><p>{report.recording_count} held-out recordings · {new Date(report.created_utc).toLocaleString()} · ±1 s on both boundaries</p></div><a className="button" href={`/api/admin/benchmarks/reports/${report.report_id}/download`} download>Download report JSON</a></div>
      <div className="benchmark-table-wrap"><table className="benchmark-table"><caption>Rally metrics pooled across recordings; median uses all matched boundary errors</caption><thead><tr><th scope="col">Detector</th><th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">Median boundary error</th><th scope="col">TP / FP / FN</th></tr></thead><tbody>{methods.map(([key, label]) => <tr key={key}><th scope="row">{label}</th><MetricCells metrics={report[`${key}_events`]}/></tr>)}</tbody></table></div>
      <p className="helper">Precision: real matches / predictions. Recall: real matches / labeled rallies. Boundary error uses matched rallies only; a dash means no usable denominator or no matched boundaries. {report.boundary_semantics}</p>
      {report.warnings.map((warning, i) => <p className="benchmark-warning" key={i}>{warning}</p>)}
      <div className="benchmark-operations"><div><small>Measured on 24–26 min sources</small><strong>{seconds(report.operations.median_measured_25_minute_seconds)}</strong><span>{report.operations.measured_25_minute_count} measured recording(s)</span></div><div><small>Estimated for 25 minutes</small><strong>{seconds(report.operations.estimated_25_minute_seconds)}</strong><span>Scaled from {report.operations.timed_recording_count} timed recording(s)</span></div><div><small>Typical manual corrections</small><strong>{decimal(report.operations.median_corrections_per_recording, 1)}</strong><span>Median across {report.operations.correction_recording_count} recording(s)</span></div></div>
      <p className="helper">{report.runtime_definition} Older analyses have no timing; rerun detection in a separate project copy to measure it while retaining reviewed labels.</p><p className="helper">{report.correction_definition}</p>
      <div className="benchmark-table-wrap"><table className="benchmark-table"><caption>Time and correction burden per recording</caption><thead><tr><th scope="col">Recording</th><th scope="col">Duration</th><th scope="col">Detection</th><th scope="col">Added / removed / trimmed</th><th scope="col">Total corrections</th></tr></thead><tbody>{report.operations.recordings.map(row => <tr key={row.source_id}><th scope="row">{row.title}</th><td>{timecode(row.duration_seconds)}</td><td>{seconds(row.detection_seconds)}</td><td>{row.corrections ? `${row.corrections.added} / ${row.corrections.removed} / ${row.corrections.trimmed}` : '—'}</td><td>{row.corrections?.total ?? '—'}</td></tr>)}</tbody></table></div>
      <label>Inspect a held-out recording<select value={foldId} onChange={e => setFoldId(e.target.value)}>{report.folds.map(row => <option key={row.held_out_source_id} value={row.held_out_source_id}>{row.source_title}</option>)}</select></label>
      {fold && <><div className="benchmark-table-wrap"><table className="benchmark-table"><caption>{fold.source_title} · held-out rally metrics</caption><thead><tr><th scope="col">Detector</th><th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">Median boundary error</th><th scope="col">TP / FP / FN</th></tr></thead><tbody>{methods.map(([key, label]) => <tr key={key}><th scope="row">{label}</th><MetricCells metrics={fold[`${key}_events`]}/></tr>)}</tbody></table></div><div className="benchmark-tracks" aria-label="Rally intervals across the source recording">{[['truth', 'Ground truth', fold.reference_intervals ?? []] as const, ...methods.map(([key, label]) => [key, label, fold.predicted_intervals?.[key] ?? []] as const)].map(([key, label, intervals]) => <div className="benchmark-track-row" key={key}><span>{label}</span><div className={`benchmark-track ${key}`}>{intervals.map((interval, i) => <span key={i} style={{ left: `${interval.start / fold.duration_seconds * 100}%`, width: `${(interval.end - interval.start) / fold.duration_seconds * 100}%` }} title={`${timecode(interval.start)}–${timecode(interval.end)}`}/>)}</div></div>)}<p className="helper">Source timeline · 00:00 to {timecode(fold.duration_seconds)}. Hover over a rally to see its boundaries.</p></div><details><summary>Exact source-time intervals</summary><div className="benchmark-interval-grid">{[['truth', 'Ground truth', fold.reference_intervals ?? []] as const, ...methods.map(([key, label]) => [key, label, fold.predicted_intervals?.[key] ?? []] as const)].map(([key, label, intervals]) => <div key={key}><h3>{label}</h3><ol>{intervals.map((interval, i) => <li key={i}>{interval.start.toFixed(3)} – {interval.end.toFixed(3)} s</li>)}</ol></div>)}</div></details></>}
    </section>}
  </main>;
}
