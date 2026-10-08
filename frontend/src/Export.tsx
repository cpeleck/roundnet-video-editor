import { useEffect, useRef, useState } from 'react';
import { cardUrl, pickFile, request, thumbnailUrl } from './api';
import { Icon } from './icons';
import { labelFor, timecode, type Command } from './App';
import { JobProgress } from './Setup';
import { Video } from './Video';
import { displayDimensions } from './geometry';
import type { Job, Project, QueueClip, Rally } from './types';

export function Export({ project, command, job, onJob, onError, busy }: { project: Project; command: Command; job: Job | null; onJob: (j: Job) => void; onError: (e: unknown) => void; busy: boolean }) {
  const saved = project.export_settings;
  const [mode, setMode] = useState<'full' | 'highlights'>(saved.mode === 'highlights' ? 'highlights' : 'full');
  const [view, setView] = useState<'video' | 'card'>('video');
  const [aspect, setAspect] = useState(String(saved.aspect_ratio ?? 'source'));
  const [scoreboard, setScoreboard] = useState(saved.scoreboard !== false);
  const [endCard, setEndCard] = useState(saved.include_stats !== false);
  const [duration, setDuration] = useState(Number(saved.stats_duration ?? 5));
  const [notes, setNotes] = useState(!!saved.include_notes);
  const [branding, setBranding] = useState(String(saved.overlay_path ?? ''));
  const [hardware, setHardware] = useState(saved.prefer_hardware !== false);
  const [destination, setDestination] = useState(String(saved.output_path ?? ''));
  const [overwrite, setOverwrite] = useState(false);
  const [queueSelection, setQueueSelection] = useState(0);
  const [cropMode, setCropMode] = useState(false);
  const [position, setPosition] = useState(0);
  const [previewing, setPreviewing] = useState(false);
  const [renderStarting, setRenderStarting] = useState(false);
  const video = useRef<HTMLVideoElement>(null);
  const queue = project.highlight_queue;
  const validRallies = project.rallies.filter(r => !r.rejected);
  const fullClips = validRallies.filter(r => r.enabled);
  const selectedQueue = queue[queueSelection];
  const selectedRally = mode === 'highlights' ? validRallies.find(r => r.rally_id === selectedQueue?.rally_id) : fullClips[queueSelection];
  const selectionStart = selectedQueue?.start_time ?? selectedRally?.start_time ?? 0;
  const selectionEnd = selectedQueue?.end_time ?? selectedRally?.end_time ?? project.video_duration;
  const activeJob = job?.kind === 'export' ? job : null;
  const presentation = { aspect_ratio: aspect, scoreboard, include_stats: endCard, stats_duration: duration, include_notes: notes, overlay_path: branding || null };
  const allSettings = { ...presentation, mode, prefer_hardware: hardware, output_path: destination };
  const autosave = (patch: Record<string, unknown>) => command('export_settings', { settings: { ...allSettings, ...patch } });
  const changeMode = async (value: 'full' | 'highlights') => {
    setMode(value); setQueueSelection(0);
    const result = await autosave({ mode: value });
    if (result && value === 'highlights' && !queue.length) await command('highlight_queue', { clips: validRallies.filter(r => r.starred).map(r => ({ rally_id: r.rally_id })) });
  };
  const setQueue = (clips: QueueClip[]) => command('highlight_queue', { clips });
  const move = (index: number, direction: number) => { const clips = [...queue]; [clips[index], clips[index + direction]] = [clips[index + direction], clips[index]]; setQueueSelection(index + direction); void setQueue(clips); };
  const browseDestination = async () => { try { const path = await pickFile('destination'); if (path) { setDestination(path); await autosave({ output_path: path }); } } catch (e) { onError(e); } };
  const render = async () => {
    setRenderStarting(true);
    try {
      const savedProject = await autosave({}); if (!savedProject) return;
      const value = await request<Job>(`/projects/${project.project_id}/export`, { mode, output_path: destination, options: presentation, queue: mode === 'highlights' ? queue : undefined, prefer_hardware: hardware, overwrite, expected_revision: savedProject.revision });
      onJob(value);
    } catch (e) { onError(e); } finally { setRenderStarting(false); }
  };
  const seekSelection = () => { const start = mode === 'highlights' ? selectionStart : selectedRally?.start_time ?? 0; setPosition(start); if (video.current) { video.current.currentTime = start; if (previewing) void video.current.play().catch(() => setPreviewing(false)); else video.current.pause(); } };
  useEffect(() => { seekSelection(); }, [selectedRally?.rally_id, queueSelection, mode, selectedQueue?.start_time]);
  const [displayWidth, displayHeight] = displayDimensions(project);
  const ratio = aspect === 'source' ? displayWidth / displayHeight : aspect === '16:9' ? 16 / 9 : aspect === '9:16' ? 9 / 16 : 1;
  const originalRatio = displayWidth / displayHeight;
  const assignment = project.timeline.find(t => t.rally_id === selectedRally?.rally_id);
  const totalDuration = mode === 'highlights' ? queue.reduce((sum, clip) => { const r = project.rallies.find(r => r.rally_id === clip.rally_id)!; return sum + (clip.end_time ?? r.end_time) - (clip.start_time ?? r.start_time); }, 0) : fullClips.reduce((sum, r) => sum + r.end_time - r.start_time, 0);
  const selectedKeys = mode === 'highlights' ? selectedQueue?.crop_keyframes ?? selectedRally?.crop_keyframes ?? [] : selectedRally?.crop_keyframes ?? [];
  // Match the exporter's smoothstep interpolation between source-time crop centers.
  const prior = [...selectedKeys].reverse().find(k => k.time <= position) ?? selectedKeys[0];
  const next = selectedKeys.find(k => k.time >= position) ?? selectedKeys[selectedKeys.length - 1];
  const mix = prior && next && next.time !== prior.time ? (position - prior.time) / (next.time - prior.time) : 0;
  const eased = mix * mix * (3 - 2 * mix);
  const center = prior ? { x: prior.x + ((next?.x ?? prior.x) - prior.x) * eased, y: prior.y + ((next?.y ?? prior.y) - prior.y) * eased } : { x: .5, y: .5 };
  const horizontalScale = originalRatio > ratio ? originalRatio / ratio : 1;
  const verticalScale = originalRatio < ratio ? ratio / originalRatio : 1;
  const cropLeft = Math.max(0, Math.min(horizontalScale - 1, center.x * horizontalScale - .5));
  const cropTop = Math.max(0, Math.min(verticalScale - 1, center.y * verticalScale - .5));
  return <main className="export-page"><div className="section-heading"><div><span className="eyebrow">SHARE THE GAME</span><h1>The final cut.<br/><span>Make it yours.</span></h1><p>Your score, your players, your best moments.</p></div><div className="segmented-control">{[['full', 'Full match'], ['highlights', 'Highlights']].map(([value, label]) => <button key={value} disabled={busy} className={mode === value ? 'active' : ''} onClick={() => void changeMode(value as 'full' | 'highlights')}>{label}</button>)}</div></div>
    {mode === 'highlights' && <section className="highlight-builder"><div className="section-heading compact"><div><h2>Choose your moments</h2><p>Stars start your shortlist. Queue order and reel trims are saved separately.</p></div><span className="count-pill">{queue.length} selected</span></div><div className="highlight-shortlist">{validRallies.map(r => <label key={r.rally_id} className={`shortlist-clip ${queue.some(c => c.rally_id === r.rally_id) ? 'selected' : ''}`}><input type="checkbox" disabled={busy} checked={queue.some(c => c.rally_id === r.rally_id)} onChange={e => { setQueueSelection(0); void setQueue(e.target.checked ? [...queue, { rally_id: r.rally_id }] : queue.filter(c => c.rally_id !== r.rally_id)); }}/><img src={thumbnailUrl(project.project_id, r.start_time, 240)} alt="" loading="lazy"/><span><strong>Clip {project.rallies.indexOf(r) + 1}{r.starred && ' ★'}</strong><small>{labelFor(r.classification.kind) || 'Needs outcome'}</small></span></label>)}</div>{!validRallies.length && <p className="helper">Add some clips in Review to build a reel.</p>}</section>}
    <div className="export-layout"><section className="export-preview"><div className="preview-tabs"><div className="filter-tabs"><button className={view === 'video' ? 'active' : ''} onClick={() => setView('video')}>Video preview</button><button className={view === 'card' ? 'active' : ''} onClick={() => setView('card')}>End card</button></div><span>{view === 'card' ? `${duration}s on screen` : `${mode === 'highlights' ? queue.length : fullClips.length} clips · ${timecode(totalDuration + (endCard ? duration : 0))}`}</span></div>
      <div className="export-preview-stage"><div className={`export-frame ${cropMode ? 'crop-mode' : ''}`} style={{ aspectRatio: ratio }}>
        {view === 'card' ? <img className="end-card-image" src={cardUrl(project.project_id, project.revision, aspect)} alt="Rendered end card matching the final video"/> : <><div className="export-video-crop" style={{ width: `${horizontalScale * 100}%`, height: `${verticalScale * 100}%`, left: `${-cropLeft * 100}%`, top: `${-cropTop * 100}%` }}><Video project={project} videoRef={video} onLoaded={seekSelection} onPlay={() => setPreviewing(true)} onPause={() => setPreviewing(false)} onTime={() => { const current = video.current?.currentTime ?? 0; setPosition(current); const end = mode === 'highlights' ? selectionEnd : selectedRally?.end_time ?? project.video_duration; if (current >= end && previewing) { const length = mode === 'highlights' ? queue.length : fullClips.length; if (queueSelection < length - 1) setQueueSelection(queueSelection + 1); else video.current?.pause(); } }} onClick={e => { if (!cropMode || !selectedRally) return; const bounds = e.currentTarget.getBoundingClientRect(); const key = { time: position, x: Math.max(0, Math.min(1, (e.clientX - bounds.left) / bounds.width)), y: Math.max(0, Math.min(1, (e.clientY - bounds.top) / bounds.height)) }; const crop_keyframes = [...selectedKeys.filter(k => Math.abs(k.time - position) > .001), key].sort((a, b) => a.time - b.time); if (mode === 'highlights') void setQueue(queue.map((clip, index) => index === queueSelection ? { ...clip, crop_keyframes } : clip)); else void command('crop', { rally_id: selectedRally.rally_id, crop_keyframes }); setCropMode(false); }}/></div>{scoreboard && assignment && <div className="preview-scoreboard"><span>{project.match_settings.team_a}<b>{assignment.score_before[0]}</b></span><span>{project.match_settings.team_b}<b>{assignment.score_before[1]}</b></span></div>}{notes && selectedRally?.note && <div className="preview-caption">{selectedRally.note}</div>}{branding && <img className="preview-branding" src={`/api/projects/${project.project_id}/branding`} alt="Branding overlay"/>}{cropMode && <div className="crop-hint">Click a framing center at {timecode(position)}</div>}</>}
      </div></div>
      {view === 'video' && <div className="playback-bar"><button className="play-button" aria-label={previewing ? 'Pause preview' : 'Play preview'} onClick={() => { if (previewing) video.current?.pause(); else void video.current?.play().catch(() => setPreviewing(false)); }}><Icon name={previewing ? 'pause' : 'play'}/></button><span className="time-readout">{timecode(position)}</span><input type="range" min={mode === 'highlights' ? selectionStart : selectedRally?.start_time ?? 0} max={mode === 'highlights' ? selectionEnd : selectedRally?.end_time ?? project.video_duration} step="0.05" value={position} aria-label="Preview clip position" onChange={e => { if (video.current) video.current.currentTime = Number(e.target.value); setPosition(Number(e.target.value)); }}/><button className="button small" disabled={busy || !selectedRally} onClick={() => setCropMode(!cropMode)}>{cropMode ? 'Cancel' : 'Set crop center'}</button></div>}
      <p className="helper preview-note">End-card preview uses the final Python renderer. Video preview shows your framing and the score entering each source clip; final overlays render with FFmpeg.</p>
      {mode === 'highlights' ? <div className="reel-queue"><div className="section-heading compact"><h2>Your reel queue</h2><span>Reorder with the arrows</span></div>{queue.map((clip, index) => { const r = project.rallies.find(r => r.rally_id === clip.rally_id)!; return <div className={`queue-row ${index === queueSelection ? 'selected' : ''}`} key={clip.rally_id}><button className="queue-select" onClick={() => { setQueueSelection(index); setView('video'); }}><span>{String(index + 1).padStart(2, '0')}</span><img src={thumbnailUrl(project.project_id, clip.start_time ?? r.start_time, 160)} alt=""/><div><strong>Clip {project.rallies.indexOf(r) + 1}</strong><small>{labelFor(r.classification.kind) || 'Unclassified'} · {timecode((clip.end_time ?? r.end_time) - (clip.start_time ?? r.start_time))}</small></div></button><button className="icon-button" aria-label={`Move clip ${index + 1} up`} disabled={busy || index === 0} onClick={() => move(index, -1)}><Icon name="arrow" style={{ transform: 'rotate(-90deg)' }}/></button><button className="icon-button" aria-label={`Move clip ${index + 1} down`} disabled={busy || index === queue.length - 1} onClick={() => move(index, 1)}><Icon name="arrow" style={{ transform: 'rotate(90deg)' }}/></button><button className="icon-button" aria-label={`Remove clip ${index + 1} from reel`} disabled={busy} onClick={() => { setQueueSelection(0); void setQueue(queue.filter((_, i) => i !== index)); }}><Icon name="close"/></button></div>; })}{selectedRally && selectedQueue && <ReelTrim key={`${selectedQueue.rally_id}-${selectedQueue.start_time}-${selectedQueue.end_time}`} clip={selectedQueue} rally={selectedRally} videoDuration={project.video_duration} busy={busy} onSave={clip => setQueue(queue.map((c, i) => i === queueSelection ? clip : c))}/>}</div> : <div className="full-preview-list">{fullClips.map((r, index) => <button key={r.rally_id} className={index === queueSelection ? 'active' : ''} onClick={() => { setQueueSelection(index); setView('video'); }}>Clip {project.rallies.indexOf(r) + 1}</button>)}</div>}
    </section>
    <aside className="export-controls"><h2>Make it ready to share.</h2><label>Framing<select value={aspect} disabled={busy} onChange={e => { setAspect(e.target.value); void autosave({ aspect_ratio: e.target.value }); }}>{[['source', 'Original recording'], ['16:9', 'Landscape · 16:9'], ['9:16', 'Portrait · 9:16'], ['1:1', 'Square · 1:1']].map(([value, title]) => <option value={value} key={value}>{title}</option>)}</select></label><label className="toggle-row"><span><strong>Scoreboard</strong><small>Score before every included point</small></span><input type="checkbox" role="switch" checked={scoreboard} disabled={busy} onChange={e => { setScoreboard(e.target.checked); void autosave({ scoreboard: e.target.checked }); }}/></label><label className="toggle-row"><span><strong>Player end card</strong><small>Updates after every point · score, stats + RPR</small></span><input type="checkbox" role="switch" checked={endCard} disabled={busy} onChange={e => { setEndCard(e.target.checked); void autosave({ include_stats: e.target.checked }); }}/></label>{endCard && <label>End-card duration <span className="input-suffix">{duration} seconds</span><input type="range" min="1" max="30" step="1" value={duration} disabled={busy} onChange={e => { setDuration(Number(e.target.value)); setView('card'); }} onPointerUp={() => void autosave({ stats_duration: duration })} onKeyUp={() => void autosave({ stats_duration: duration })}/><button className="text-button" onClick={() => setView('card')}>Preview end card<Icon name="arrow" size={14}/></button></label>}<label>Save video to<input value={destination} disabled={busy} onChange={e => setDestination(e.target.value)} onBlur={e => { if (!(e.relatedTarget as HTMLElement)?.closest('.button.primary') && destination !== saved.output_path) void autosave({ output_path: destination }); }} placeholder="/path/to/replay.mp4"/><button className="text-button" disabled={busy} onClick={() => void browseDestination()}><Icon name="folder" size={15}/>Choose destination</button></label>
      <details><summary>More options</summary><label className="checkbox"><input type="checkbox" checked={notes} disabled={busy} onChange={e => { setNotes(e.target.checked); void autosave({ include_notes: e.target.checked }); }}/>Include clip captions</label><label>Branding PNG path<input value={branding} disabled={busy} placeholder="/path/to/logo.png" onChange={e => setBranding(e.target.value)} onBlur={e => { if (!(e.relatedTarget as HTMLElement)?.closest('.button.primary')) void autosave({ overlay_path: branding || null }); }}/></label><label>Encoder<select value={hardware ? 'auto' : 'software'} disabled={busy} onChange={e => { const value = e.target.value === 'auto'; setHardware(value); void autosave({ prefer_hardware: value }); }}><option value="auto">Automatic · use hardware when available</option><option value="software">Software H.264</option></select></label><label className="checkbox"><input type="checkbox" checked={overwrite} onChange={e => setOverwrite(e.target.checked)}/>Replace an existing output file</label><CorrectionLabels project={project} command={command} busy={busy} onError={onError}/></details>
      {project.statistics.coverage.unresolved > 0 && <div className="provisional-notice"><strong>{project.statistics.coverage.unresolved} outcomes still missing</strong><p>The end card will show the recorded score and unknown totals where needed.</p><button className="text-button" disabled={busy} onClick={() => void command('stage', { stage: 'review' })}>Continue reviewing<Icon name="arrow" size={14}/></button></div>}<div className="render-summary"><span>{mode === 'highlights' ? 'Highlight reel' : 'Full match'}<strong>{timecode(totalDuration + (endCard ? duration : 0))}</strong></span><small>MP4 · H.264 · Original audio when available</small></div><button className="button primary wide" disabled={busy || renderStarting || !destination.trim() || !(mode === 'highlights' ? queue.length : fullClips.length)} onClick={() => void render()}><Icon name="export"/>{renderStarting ? 'Starting export…' : 'Export video'}</button>{activeJob && <><JobProgress job={activeJob} onError={onError}/>{activeJob.status === 'succeeded' && <div className="export-complete"><Icon name="check"/><strong>Your replay is ready.</strong><span>{activeJob.result?.output_path}</span><a className="button wide" href={`/api/jobs/${activeJob.id}/output`} download>Download MP4</a></div>}</>}
    </aside></div></main>;
}
function ReelTrim({ clip, rally, videoDuration, busy, onSave }: { clip: QueueClip; rally: Rally; videoDuration: number; busy: boolean; onSave: (clip: QueueClip) => Promise<unknown> }) {
  const [start, setStart] = useState(clip.start_time ?? rally.start_time); const [end, setEnd] = useState(clip.end_time ?? rally.end_time);
  return <details className="reel-trim"><summary>Trim selected reel clip <span>Match point stays unchanged</span></summary><form onSubmit={e => { e.preventDefault(); void onSave({ ...clip, start_time: start, end_time: end }); }}><div className="row"><label>In (source seconds)<input type="number" min="0" max={videoDuration} step="0.01" value={start} onChange={e => setStart(Number(e.target.value))}/></label><label>Out (source seconds)<input type="number" min="0" max={videoDuration} step="0.01" value={end} onChange={e => setEnd(Number(e.target.value))}/></label><button className="button small" type="submit" disabled={busy || end <= start || end > videoDuration}>Save reel trim</button></div></form></details>;
}
function CorrectionLabels({ project, command, busy, onError }: { project: Project; command: Command; busy: boolean; onError: (e: unknown) => void }) {
  const [preparing, setPreparing] = useState(false);
  const [download, setDownload] = useState<{ download_url: string; sample_count: number; features_available: boolean } | null>(null);
  const review = project as Project & { complete?: boolean; complete_review_confirmed?: boolean };
  const confirmed = !!review.complete && !!review.complete_review_confirmed;
  const prepare = async () => {
    setPreparing(true); setDownload(null);
    try {
      const result = await request<{ download_url: string; sample_count: number; features_available: boolean }>(`/projects/${project.project_id}/correction-labels`, { expected_revision: project.revision, full_review_confirmed: confirmed });
      setDownload(result);
      const link = document.createElement('a');
      link.href = result.download_url; link.download = 'roundnet-correction-labels.zip'; link.click();
    } catch (e) { onError(e); } finally { setPreparing(false); }
  };
  return <details><summary>Correction-label output</summary><p className="helper">Download detector corrections for local training. Unreviewed background stays unknown unless you checked the whole recording for missed rallies.</p><label className="checkbox"><input type="checkbox" disabled={busy || preparing} checked={confirmed} onChange={e => { setDownload(null); void command('complete_review', { confirmed: e.target.checked }); }}/>I checked the entire video for missed rallies</label><button className="button small" disabled={busy || preparing} onClick={() => void prepare()}>{preparing ? 'Preparing labels…' : 'Download correction labels'}</button>{download && <p className="helper">{download.features_available ? `${download.sample_count} aligned feature samples included.` : 'Source-time JSON included. Run detection to include training features.'} <a href={download.download_url} download="roundnet-correction-labels.zip">Download ZIP again</a></p>}</details>;
}
