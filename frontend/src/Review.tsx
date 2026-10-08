import { useEffect, useRef, useState } from 'react';
import { cardUrl, thumbnailUrl } from './api';
import { Icon } from './icons';
import { labelFor, timecode, type Command } from './App';
import { Video } from './Video';
import { Modal } from './Modal';
import { displayDimensions } from './geometry';
import type { PointStats, Project, QuickLog, Rally, Touch } from './types';
import { emptyQuickLog, QuickLogEditor } from './QuickLogEditor';

const groups = [
  { title: 'Serving team wins', className: 'serving', actions: [['ace', 'Ace', 'Unreturnable serve', '1'], ['service_break', 'Service Break', 'Service pressure wins', '2'], ['defensive_hold', 'Defensive Hold', 'Serving team converts defense', '3']] },
  { title: 'Receiving team wins', className: 'receiving', actions: [['double_fault', 'Double Fault', 'Two service errors', '4'], ['sideout', 'Sideout', 'Clean return wins', '5'], ['defensive_break', 'Defensive Break', 'Receiving team converts defense', '6']] },
  { title: 'Other outcomes', className: 'other', actions: [['error', 'Error', 'Choose the player responsible', '7'], ['redo', 'Redo', 'No point · same serving pair', '8']] },
];
export function Review({ project, command, busy, shortcuts, onStats }: { project: Project; command: Command; busy: boolean; shortcuts: boolean; onStats: () => void }) {
  const video = useRef<HTMLVideoElement>(null);
  const outcomeFocus = useRef<HTMLElement>(null);
  const playRange = useRef<{ start: number; end: number } | null>(null);
  const [position, setPosition] = useState(project.position || 0);
  const [playing, setPlaying] = useState(false);
  const [markStart, setMarkStart] = useState<number | null>(null);
  const [guidedKind, setGuidedKind] = useState<string | null>(null);
  const [playbackMessage, setPlaybackMessage] = useState('');
  const [markerError, setMarkerError] = useState('');
  const managedSelection = useRef<string | null>(null);
  const playbackGeneration = useRef(0);
  const continuous = useRef(true);
  const [touchOpen, setTouchOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [cropMode, setCropMode] = useState(false);
  const [filter, setFilter] = useState('all');
  const [loop, setLoop] = useState(false);
  const rally = project.rallies[project.selected];
  const assignment = project.timeline.find(t => t.rally_id === rally?.rally_id);
  const names = Object.fromEntries(project.match_settings.players.map(p => [p.player_id, p.name]));
  const firstMissing = project.rallies.find(r => !r.rejected && !r.classification.kind && !r.winner && r.outcome !== 'Replay / no point');
  const stats = project.statistics;
  const [displayWidth, displayHeight] = displayDimensions(project);
  const visible = project.rallies.filter(r => filter === 'all' || filter === 'pending' && !r.rejected && !r.classification.kind && !r.winner || filter === 'stars' && r.starred || filter === 'excluded' && (!r.enabled || r.rejected));
  const currentTime = () => video.current?.currentTime ?? position;
  const seek = (seconds: number) => { const value = Math.max(0, Math.min(project.video_duration, seconds)); if (video.current) video.current.currentTime = value; setPosition(value); };
  const startPlayback = async () => {
    setPlaybackMessage('');
    if (!video.current || currentTime() >= project.video_duration) { setPlaying(false); return; }
    try { await video.current.play(); } catch { setPlaying(false); setPlaybackMessage('Playback is paused. Click Resume to continue.'); }
  };
  const select = async (r: Rally) => {
    setGuidedKind(null); setMenuOpen(false); setCropMode(false);
    playbackGeneration.current++; video.current?.pause();
    managedSelection.current = '*';
    const value = await command('select', { rally_id: r.rally_id });
    if (!value) managedSelection.current = null;
    if (value) { managedSelection.current = r.rally_id; continuous.current = false; playRange.current = { start: r.start_time, end: r.end_time }; seek(r.start_time); }
  };
  const adjacent = (direction: number) => { const next = project.rallies[Math.max(0, Math.min(project.rallies.length - 1, project.selected + direction))]; if (next) void select(next); };
  const commitPoint = async (kind: string, quick_log?: QuickLog) => {
    if (!rally || busy) return;
    const end = rally.end_time;
    const generation = ++playbackGeneration.current;
    managedSelection.current = '*';
    const value = await command('classify', { rally_id: rally.rally_id, kind, ...(kind === 'error' ? { player_id: quick_log?.error?.player_id ?? rally.classification.player_id } : {}), ...(quick_log ? { quick_log } : {}) });
    if (!value) managedSelection.current = null;
    if (value && generation === playbackGeneration.current) {
      managedSelection.current = value.rallies[value.selected]?.rally_id ?? null;
      setGuidedKind(null); playRange.current = null; continuous.current = true;
      seek(end); outcomeFocus.current?.focus({ preventScroll: true }); await startPlayback();
    }
  };
  const classify = async (kind: string) => {
    if (!rally || busy || markStart !== null) return;
    if (rally.classification.kind === kind && rally.point_stats.events?.length && !rally.point_stats.quick_log) {
      await commitPoint(kind); return;
    }
    if (['defensive_hold', 'defensive_break', 'error'].includes(kind)) {
      video.current?.pause(); setGuidedKind(kind); return;
    }
    await commitPoint(kind, rally.classification.kind === kind ? rally.point_stats.quick_log ?? emptyQuickLog() : emptyQuickLog());
  };
  const markBegin = () => { if (busy || guidedKind || touchOpen) return; setMarkerError(''); setMarkStart(currentTime()); playRange.current = null; continuous.current = true; };
  const markEnd = async () => {
    if (markStart === null || busy) return;
    const end = currentTime();
    if (end <= markStart) { setMarkerError('The clip end must be after its start. Seek forward and mark the end again.'); return; }
    video.current?.pause(); playbackGeneration.current++; setMarkerError('');
    managedSelection.current = '*';
    const value = await command('add', { start_time: markStart, end_time: end });
    if (!value) managedSelection.current = null;
    if (value) { managedSelection.current = value.rallies[value.selected]?.rally_id ?? null; setMarkStart(null); continuous.current = false; playRange.current = null; seek(end); }
  };
  const replay = () => { if (!rally) return; continuous.current = false; playRange.current = { start: rally.start_time, end: rally.end_time }; seek(rally.start_time); void startPlayback(); };
  const play = () => {
    if (!video.current) return;
    if (!video.current.paused) video.current.pause();
    else if (!continuous.current && rally && markStart === null) {
      playRange.current = { start: rally.start_time, end: rally.end_time };
      if (currentTime() < rally.start_time || currentTime() >= rally.end_time) seek(rally.start_time);
      void startPlayback();
    } else void startPlayback();
  };
  useEffect(() => {
    if (managedSelection.current === '*') return;
    if (managedSelection.current === rally?.rally_id) { managedSelection.current = null; return; }
    playbackGeneration.current++; video.current?.pause(); playRange.current = null;
    seek(project.position ?? rally?.start_time ?? 0);
  }, [rally?.rally_id]);
  useEffect(() => {
    if (busy || touchOpen || guidedKind || !shortcuts || Math.abs(position - project.position) < .05) return;
    const timer = window.setTimeout(() => { void command('position', { position }); }, 1500);
    return () => clearTimeout(timer);
  }, [position, project.position, busy, touchOpen, guidedKind, shortcuts, command]);
  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      if (!shortcuts || touchOpen || busy || event.metaKey || event.ctrlKey || event.altKey || (event.target instanceof Element && event.target.matches('input,textarea,select,[contenteditable=true]'))) return;
      const key = event.key.toLowerCase();
      if (key === 'arrowleft' || key === 'arrowright') { event.preventDefault(); seek(currentTime() + (key === 'arrowleft' ? -5 : 5)); return; }
      if (key === ' ' && !(event.target instanceof Element && event.target.closest('button'))) { event.preventDefault(); play(); return; }
      if (guidedKind) { if (key === 'escape') setGuidedKind(null); return; }
      const kind = groups.flatMap(g => g.actions).find(a => a[3] === event.key)?.[0];
      if (kind) { event.preventDefault(); void classify(kind); return; }
      if (key === '[') adjacent(-1); else if (key === ']') adjacent(1);
      else if (key === 'w' && markStart === null) markBegin(); else if (key === 'x') void markEnd();
      else if (key === 'f' && rally) void command('star', { rally_id: rally.rally_id, starred: !rally.starred });
      else if (key === 'escape') { setMarkStart(null); setMarkerError(''); setCropMode(false); }
    };
    window.addEventListener('keydown', listener); return () => window.removeEventListener('keydown', listener);
  });
  const moreCommand = (type: string, values: Record<string, unknown> = {}) => { if (rally) void command(type, { rally_id: rally.rally_id, ...values }); setMenuOpen(false); };
  return <main className="review-page"><div className="review-heading"><div><span className="eyebrow">MAKE THE CALL</span><h1>Every rally tells a story.</h1></div><div className="review-coverage"><span><i/>{stats.coverage.classified ?? 0} outcomes logged</span><span>{stats.coverage.unresolved ?? 0} still to classify</span></div></div>
    <div className={`editor-layout ${guidedKind ? 'logging' : ''}`}><section className="player-column"><div className={`video-stage ${cropMode ? 'crop-mode' : ''}`} style={{ aspectRatio: `${displayWidth}/${displayHeight}` }}><Video project={project} videoRef={video} onLoaded={() => { setPlaying(false); seek(project.position ?? rally?.start_time ?? 0); }} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => { setPlaying(false); playRange.current = null; }} onTime={() => { const value = video.current?.currentTime ?? 0; setPosition(value); if (playRange.current && value >= playRange.current.end && playing) { if (loop) seek(playRange.current.start); else video.current?.pause(); } }} onClick={e => { if (!cropMode || !rally) return; const rect = e.currentTarget.getBoundingClientRect(); const keyframe = { time: position, x: (e.clientX - rect.left) / rect.width, y: (e.clientY - rect.top) / rect.height }; const keys = [...rally.crop_keyframes.filter(k => Math.abs(k.time - position) > .001), keyframe].sort((a, b) => a.time - b.time); void command('crop', { rally_id: rally.rally_id, crop_keyframes: keys }); setCropMode(false); }}/>{rally && <div className="video-corner-label">CLIP {project.selected + 1}<span>{timecode(rally.end_time - rally.start_time)}</span></div>}{cropMode && <div className="crop-hint">Click the framing center at {timecode(position)}</div>}</div>
      <div className="playback-bar"><button className="play-button" onClick={play} aria-label={playing ? 'Pause video' : 'Play video'}><Icon name={playing ? 'pause' : 'play'} size={19}/></button><button className="button small seek-button" aria-label="Back five seconds" disabled={busy} onClick={() => seek(currentTime() - 5)}>−5s</button><button className="button small seek-button" aria-label="Forward five seconds" disabled={busy} onClick={() => seek(currentTime() + 5)}>+5s</button><span className="time-readout">{timecode(position)} <i>/</i> {timecode(project.video_duration)}</span><input disabled={busy} aria-label="Source playback position" type="range" min="0" max={project.video_duration} step="0.05" value={position} onChange={e => seek(Number(e.target.value))}/><label className="loop-toggle"><input type="checkbox" checked={loop} onChange={e => setLoop(e.target.checked)}/>Loop</label></div>
      <div className="clip-toolbar"><div className="row"><button className="button small" disabled={busy || project.selected <= 0} onClick={() => adjacent(-1)}><Icon name="back"/>Previous</button><button className="button small" disabled={busy || project.selected >= project.rallies.length - 1} onClick={() => adjacent(1)}>Next<Icon name="arrow"/></button></div><div className="row"><button className={`button small ${rally?.starred ? 'starred' : ''}`} disabled={busy || !rally} onClick={() => rally && void command('star', { rally_id: rally.rally_id, starred: !rally.starred })}><Icon name="star"/>{rally?.starred ? 'Starred' : 'Star'}</button><button className="button small" disabled={busy || !!guidedKind || touchOpen} onClick={() => markStart === null ? markBegin() : void markEnd()}><Icon name="plus"/>{markStart === null ? 'Mark start' : 'Mark end'}</button></div></div>
      {markStart !== null && <div className="notice">Start marked at {timecode(markStart)}. Seek to the end, then click Mark end (X).<button onClick={() => { setMarkStart(null); setMarkerError(''); }}>Cancel</button></div>}
      {markerError && <div className="notice warning-text" role="alert">{markerError}</div>}
      {playbackMessage && <div className="notice" role="status">{playbackMessage}<button onClick={() => void startPlayback()}>Resume</button></div>}
      {project.explanation && <div className="classification-receipt" role="status"><Icon name="check"/><span>{project.explanation}</span><button disabled={busy || !project.can_undo} onClick={() => void command('undo')}>Undo</button></div>}
      {!project.rallies.length && <div className="empty-editor"><h2>Your first rally starts here.</h2><p>Play the recording. Press <kbd>W</kbd> at the start of a rally and <kbd>X</kbd> at the end. Then choose its outcome.</p></div>}
    </section>
    <aside className="outcome-panel" ref={outcomeFocus} tabIndex={-1} aria-label="Selected clip outcome">{rally && assignment ? <><div className="selected-clip-meta"><span>Clip {project.selected + 1} of {project.rallies.length}</span><span className={`status-pill ${rally.rejected ? 'rejected' : rally.classification.kind ? 'done' : ''}`}>{rally.rejected ? 'Not a rally' : rally.classification.kind ? labelFor(rally.classification.kind) : 'Needs outcome'}</span></div><div className="scoreboard"><span>{project.match_settings.team_a}</span><strong>{assignment.score_before[0]}<i>–</i>{assignment.score_before[1]}</strong><span>{project.match_settings.team_b}</span><small>SCORE ENTERING THIS CLIP</small></div><div className="serve-assignment"><span><i className="serving-dot"/>Server <strong>{names[assignment.server_id]}</strong></span><span>Receiver <strong>{names[assignment.receiver_id]}</strong></span></div>
      {assignment.provisional && <div className="provisional-notice"><strong>Serving order is provisional</strong><p>An earlier clip needs an outcome before this point can be scored.</p>{firstMissing && <button className="button small" onClick={() => void select(firstMissing)}>Go to missing outcome<Icon name="arrow"/></button>}</div>}
      {!guidedKind && <><h2>What happened?</h2><div className="outcome-groups">{groups.map(group => <div key={group.title} className={group.className}><div className="outcome-group-title">{group.title}</div>{group.actions.map(([kind, label, description, key]) => <button className={`outcome-button ${rally.classification.kind === kind ? 'selected' : ''}`} key={kind} disabled={busy || !!guidedKind || markStart !== null || rally.rejected || assignment.provisional} onClick={() => void classify(kind)}><span><strong>{label}</strong><small>{description}</small></span><kbd>{key}</kbd></button>)}</div>)}</div></>}
      {guidedKind && <QuickLogEditor key={`${rally.rally_id}-${guidedKind}`} project={project} rally={rally} assignment={assignment} kind={guidedKind} busy={busy} onReplay={replay} onCancel={() => setGuidedKind(null)} onSave={log => commitPoint(guidedKind, log)}/>}
      <details className="trim-details"><summary>Trim clip <span>{timecode(rally.start_time)} – {timecode(rally.end_time)}</span></summary><Trim key={`${rally.rally_id}-${rally.start_time}-${rally.end_time}`} rally={rally} duration={project.video_duration} command={command} position={position} busy={busy}/></details>
      <details className="more-details"><summary>More details <span>Touches & caption</span></summary><button className="button small wide" disabled={busy || !!guidedKind || rally.rejected} onClick={() => setTouchOpen(true)}>{rally.classification.kind?.startsWith('defensive') ? 'Add defensive touches' : 'Log touch details'}</button>{rally.point_stats.quick_log && <button className="button small wide" disabled={busy || !!guidedKind} onClick={() => { video.current?.pause(); setGuidedKind(rally.classification.kind!); }}>Edit quick log</button>}<p className="helper">Guided player details support RPR. Full touch details add receive and set quality.</p><Caption key={`${rally.rally_id}-${rally.note}`} rally={rally} command={command} busy={busy}/></details>
      <div className="clip-menu-container"><button className="button small wide" disabled={busy} onClick={() => setMenuOpen(!menuOpen)}>Clip actions<Icon name="chevron"/></button>{menuOpen && <div className="clip-menu"><button onClick={() => moreCommand('split', { time: position })} disabled={position <= rally.start_time || position >= rally.end_time}>Split at playhead</button><button disabled={project.selected >= project.rallies.length - 1} onClick={() => moreCommand('merge', { rally_ids: [rally.rally_id, project.rallies[project.selected + 1]?.rally_id] })}>Merge with next clip</button><button onClick={() => { setCropMode(true); setMenuOpen(false); }}>Add crop center at playhead</button><button onClick={() => moreCommand('crop', { crop_keyframes: [] })} disabled={!rally.crop_keyframes.length}>Clear crop centers</button><button onClick={() => moreCommand('clear_classification')} disabled={!rally.classification.kind}>Clear outcome</button><button onClick={() => moreCommand('export_enabled', { enabled: !rally.enabled })} disabled={rally.rejected}>{rally.enabled ? 'Exclude from exported video' : 'Include in exported video'}</button><button className="danger" onClick={() => moreCommand('reject', { rejected: !rally.rejected })}>{rally.rejected ? 'Restore as a rally' : 'Not a rally'}</button></div>}</div>
    </> : <div className="outcome-empty"><Icon name="video" size={36}/><h2>Let’s find your first clip.</h2><p>Mark a start and end in the video. Your outcome buttons will appear here.</p><button className="button" onClick={() => void command('stage', { stage: 'find' })}>Find rallies automatically</button></div>}</aside></div>
    <section className="filmstrip-section"><div className="filmstrip-heading"><div><strong>Match timeline</strong><span>{project.rallies.length} clips · source timestamps</span></div><div className="filter-tabs">{[['all', 'All clips'], ['pending', 'Needs outcome'], ['stars', 'Starred'], ['excluded', 'Excluded']].map(([key, label]) => <button className={filter === key ? 'active' : ''} key={key} onClick={() => setFilter(key)}>{label}</button>)}</div></div><div className="filmstrip">{visible.map(r => <button key={r.rally_id} className={`filmstrip-clip ${r.rally_id === rally?.rally_id ? 'selected' : ''} ${r.rejected ? 'rejected' : ''}`} disabled={busy} onClick={() => void select(r)}><div><img loading="lazy" src={thumbnailUrl(project.project_id, r.start_time, 240)} alt=""/><span>{project.rallies.indexOf(r) + 1}</span>{r.starred && <Icon name="star" size={14}/>}</div><strong>{r.rejected ? 'Not a rally' : labelFor(r.classification.kind) || (r.winner ? 'Legacy point' : 'Needs outcome')}</strong><small>{timecode(r.start_time)}{!r.enabled && !r.rejected ? ' · Export excluded' : ''}</small></button>)}{!visible.length && <p className="helper">{project.rallies.length ? 'No clips match this filter.' : 'Your clips will appear here as you mark them.'}</p>}</div></section>
    <footer className="editor-footer"><span><kbd>Space</kbd>Play / pause <kbd>1–8</kbd>Outcome <kbd>← →</kbd>Seek 5s <kbd>[ ]</kbd>Previous / next <kbd>W X</kbd>Mark start / end</span><button onClick={onStats}><Icon name="stats"/>View match statistics</button></footer>
    {touchOpen && rally && assignment && <Touches project={project} rally={rally} command={command} onClose={() => setTouchOpen(false)}/>}
  </main>;
}

function Trim({ rally, duration, command, position, busy }: { rally: Rally; duration: number; command: Command; position: number; busy: boolean }) {
  const [start, setStart] = useState(rally.start_time); const [end, setEnd] = useState(rally.end_time);
  return <form onSubmit={e => { e.preventDefault(); void command('trim', { rally_id: rally.rally_id, start_time: start, end_time: end }); }}><div className="row"><label>In (seconds)<input type="number" min="0" max={duration} step="0.01" value={start} onChange={e => setStart(Number(e.target.value))}/><button type="button" className="text-button" onClick={() => setStart(position)}>Use playhead</button></label><label>Out (seconds)<input type="number" min="0" max={duration} step="0.01" value={end} onChange={e => setEnd(Number(e.target.value))}/><button type="button" className="text-button" onClick={() => setEnd(position)}>Use playhead</button></label></div><button className="button small wide" type="submit" disabled={busy || end <= start || end > duration}>Save trim</button></form>;
}
function Caption({ rally, command, busy }: { rally: Rally; command: Command; busy: boolean }) {
  const [value, setValue] = useState(rally.note);
  return <form onSubmit={e => { e.preventDefault(); void command('caption', { rally_id: rally.rally_id, note: value }); }}><label>Caption<input value={value} onChange={e => setValue(e.target.value)} placeholder="A note for this moment" maxLength={2000}/></label><button className="button small wide" disabled={busy || value === rally.note}>Save caption</button></form>;
}
const touchResults: Record<string, string[]> = { serve: ['in', 'ace', 'fault', 'rim', 'let'], receive: ['strong', 'weak', 'error', 'unknown'], set: ['strong', 'weak', 'error', 'unknown'], hit: ['auto', 'put_away', 'returned', 'error'], defense: ['auto', 'get', 'touch_not_returned', 'no_touch'] };
function Touches({ project, rally, command, onClose }: { project: Project; rally: Rally; command: Command; onClose: () => void }) {
  const assignment = project.timeline.find(a => a.rally_id === rally.rally_id)!;
  const [events, setEvents] = useState<Touch[]>(rally.point_stats.events ?? []);
  const [kind, setKind] = useState('serve'); const [result, setResult] = useState('in');
  const [tough, setTough] = useState(false); const [complete, setComplete] = useState(rally.point_stats.complete ?? false);
  const [saving, setSaving] = useState(false);
  const save = async () => { setSaving(true); const point_stats: PointStats = { version: 1, server_id: assignment.server_id, receiver_id: assignment.receiver_id, complete, events }; const saved = await command('touches', { rally_id: rally.rally_id, point_stats }); setSaving(false); if (saved) onClose(); };
  return <Modal className="touch-modal" labelledBy="touch-title" onClose={onClose}><div className="modal-heading"><div><span className="eyebrow">OPTIONAL POINT DETAILS</span><h2 id="touch-title">Who touched it next?</h2></div><button className="icon-button" onClick={onClose} aria-label="Close touch details"><Icon name="close"/></button></div><p>Choose the touch and result, then tap a player. Repeated taps keep the sequence in order. Log serve attempts first.</p><div className="row"><label>Touch<select value={kind} onChange={e => { setKind(e.target.value); setResult(touchResults[e.target.value][0]); setTough(false); }}>{Object.keys(touchResults).map(k => <option key={k} value={k}>{labelFor(k)}</option>)}</select></label><label>Result<select value={result} onChange={e => { setResult(e.target.value); setTough(false); }}>{touchResults[kind].map(k => <option key={k} value={k}>{labelFor(k)}</option>)}</select></label></div>{['receive', 'set'].includes(kind) && result === 'weak' && <label className="checkbox"><input type="checkbox" checked={tough} onChange={e => setTough(e.target.checked)}/>Costly tough touch</label>}<div className="player-picker">{project.match_settings.players.map(p => <button key={p.player_id} className={`player-button team-${p.team.toLowerCase()}`} onClick={() => setEvents([...events, { event_id: crypto.randomUUID(), player_id: p.player_id, kind, result, time: null, tough }])}>{p.name}</button>)}</div><ol className="touch-list">{events.map(e => <li key={e.event_id}><strong>{project.match_settings.players.find(p => p.player_id === e.player_id)?.name}</strong><span>{labelFor(e.kind)} · {labelFor(e.result)}{e.tough ? ' · Tough' : ''}</span></li>)}</ol><button className="button small" disabled={!events.length} onClick={() => setEvents(events.slice(0, -1))}><Icon name="undo"/>Undo touch</button><label className="checkbox"><input type="checkbox" checked={complete} onChange={e => setComplete(e.target.checked)}/>I logged every touch in this point</label><p className="helper">Drafts can be saved. Inconsistent logs are flagged in Statistics and do not enter touch totals.</p><div className="modal-actions"><button className="button" onClick={onClose}>Cancel</button><button className="button primary" disabled={saving} onClick={() => void save()}>Save touch details</button></div></Modal>;
}

export function Statistics({ project, command, onClose }: { project: Project; command: Command; onClose: () => void }) {
  const report = project.statistics; const coverage = report.coverage;
  const detailKnown = coverage.complete === coverage.points - coverage.unresolved && coverage.complete > 0 && !coverage.invalid && !coverage.partial;
  const metrics: [string, string, string?, string?][] = [['aces', 'Aces'], ['aced', 'Aced'], ['double_faults', 'Double faults'], ['errors', 'Errors'], ['serve_pct', 'Serve %', 'serves_in', 'serve_attempts'], ['put_away_pct', 'Put-away %', 'put_aways', 'put_aways+hits_returned+hit_errors'], ['strong_set_pct', 'Strong set %', 'strong_sets', 'strong_sets+weak_sets+set_errors'], ['defensive_gets', 'Defensive gets'], ['rpr', report.rpr_final ? 'Original RPR' : 'Original RPR so far']];
  return <Modal className="stats-modal" labelledBy="stats-title" onClose={onClose}><div className="modal-heading"><div><span className="eyebrow">THE GAME, BY THE NUMBERS</span><h2 id="stats-title">Match statistics</h2></div><button className="icon-button" aria-label="Close statistics" onClick={onClose}><Icon name="close"/></button></div><div className="coverage-grid"><div><small>OUTCOME COVERAGE</small><strong>{Math.max(0, coverage.points - coverage.unresolved)} / {coverage.points}</strong><span>{coverage.unresolved ? `${coverage.unresolved} points still need an outcome` : 'Known outcomes are up to date'}</span></div><div><small>RPR COVERAGE</small><strong>{coverage.rpr_complete ?? coverage.complete} / {coverage.points}</strong><span>{coverage.partial} drafts · {coverage.invalid} logs need attention</span></div></div><img className="stats-card-preview" src={cardUrl(project.project_id, project.revision, '16:9')} alt="Four-player score and statistics card generated by the export renderer"/><p className="helper">{`Quick logs use the shown serve and player-role defaults for RPR. Receive and set quality stay unknown until graded. Full touch-quality coverage: ${coverage.touch_quality_complete ?? 0} / ${coverage.points}.`}</p><div className="stats-table-wrap"><table className="stats-table"><thead><tr><th>Player statistics</th>{report.players.map(p => <th key={p.player_id}>{p.name}<small>{p.team === 'A' ? project.match_settings.team_a : project.match_settings.team_b}</small></th>)}</tr></thead><tbody>{metrics.map(([key, title, numerator, denominator]) => <tr key={key}><th>{title}</th>{report.players.map(p => { const raw = key === 'rpr' ? p.rpr.overall : key === 'defensive_gets' && !detailKnown ? null : p[key]; const value = typeof raw === 'number' ? key.endsWith('_pct') ? `${Math.round(raw * 100)}%` : key === 'rpr' ? raw.toFixed(1) : raw : '—'; return <td key={p.player_id}>{value}{numerator && denominator && raw != null && <small>{String(p[numerator])} / {denominator.split('+').reduce((sum, field) => sum + Number(p[field] ?? 0), 0)}</small>}</td>; })}</tr>)}</tbody></table></div><label className="checkbox full-match-confirm"><input type="checkbox" checked={!!project.match_settings.stats_complete} disabled={!!project.busy_job_id} onChange={e => void command('full_match', { confirmed: e.target.checked })}/>I confirm this recording contains every point in the match</label><p className="helper">RPR recalculates automatically after every saved point with complete, valid player evidence and serve and hit attempts. Until the full match is confirmed, it is a rating for the recorded points so far. Clip stars, captions, crops, and export exclusions do not affect eligibility.</p><details><summary>Rating components & formulas</summary><div className="stats-table-wrap"><table className="stats-table"><thead><tr><th>Original Max Model</th>{report.players.map(p => <th key={p.player_id}>{p.name}</th>)}</tr></thead><tbody>{['hitting', 'serving', 'defense', 'efficiency'].map(key => <tr key={key}><th>{labelFor(key)}</th>{report.players.map(p => <td key={p.player_id}>{p.rpr[key]?.toFixed(1) ?? '—'}</td>)}</tr>)}</tbody></table></div><p className="helper">Hitting = 20 × (1 − returned hits / hit attempts). Serving = 5.5 × aces + 15 × serve %. Defense = (unreturned touches + 0.4 × Hitting × gets) × min(1, 44 / points). Efficiency = 20 − 5 × set/hit errors − 2 × (tough touches + aced).</p></details><details><summary>Point log & data issues</summary>{report.points.map((p, i) => <div key={p.rally_id} className="point-log"><span>{i + 1} · {timecode(p.start)}</span><strong>{labelFor(p.classification) || 'Unclassified'}{p.winner ? ` · Team ${p.winner}` : ''}</strong>{p.issues.map(issue => <p className="warning-text" key={issue}>{issue}</p>)}</div>)}</details><div className="modal-actions"><a className="button" href={`/api/projects/${project.project_id}/statistics?format=json`} download>Download JSON</a><a className="button" href={`/api/projects/${project.project_id}/statistics?format=csv`} download>Download CSV</a><a className="button" href={cardUrl(project.project_id, project.revision, '16:9')} download="match-card.png">Download card</a><button className="button primary" onClick={onClose}>Back to the game</button></div></Modal>;
}
