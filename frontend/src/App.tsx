import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError, request, thumbnailUrl, pickFile, registerMediaSource } from './api';
import { Icon } from './icons';
import type { Job, Project, ProjectSummary, Stage } from './types';
import { Setup, FindClips } from './Setup';
import { Review, Statistics } from './Review';
import { Export } from './Export';
import { Modal } from './Modal';
import { createCommandQueue } from './commandQueue';

export type Command = (type: string, values?: Record<string, unknown>) => Promise<Project | undefined>;
export const timecode = (time: number) => `${Math.floor(time / 60).toString().padStart(2, '0')}:${(time % 60).toFixed(1).padStart(4, '0')}`;
export const labelFor = (kind?: string) => (kind ?? '').split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
const stages: [Stage, string][] = [['setup', 'Match setup'], ['find', 'Find clips'], ['review', 'Review'], ['export', 'Export']];

export default function App() {
  const [project, setProject] = useState<Project | null>(null);
  const current = useRef<Project | null>(null);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [screen, setScreen] = useState<'projects' | 'workspace' | 'new'>('projects');
  const [loading, setLoading] = useState(true);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const [statsOpen, setStatsOpen] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [openPath, setOpenPath] = useState('');
  const [recovery, setRecovery] = useState<{ projectPath?: string; videoPath: string; changed: boolean; relink: boolean } | null>(null);
  const [savePath, setSavePath] = useState<string | null>(null);
  const update = useCallback((value: Project) => { registerMediaSource(value); current.current = value; setProject(value); }, []);
  const fail = useCallback((e: unknown) => setError(e instanceof Error ? e.message : String(e)), []);
  const refreshProjects = useCallback(async () => {
    setLoading(true);
    try { const data = await request<{ projects: ProjectSummary[] }>('/projects'); data.projects.forEach(registerMediaSource); setProjects(data.projects); }
    catch (e) { fail(e); } finally { setLoading(false); }
  }, [fail]);
  useEffect(() => { void refreshProjects(); }, [refreshProjects]);
  useEffect(() => { window.scrollTo({ top: 0 }); }, [screen, project?.project_id, project?.stage]);
  const queue = useRef<ReturnType<typeof createCommandQueue<Project>> | null>(null);
  if (!queue.current) queue.current = createCommandQueue<Project>(async (type, values, projectId) => {
    if (!current.current || current.current.project_id !== projectId) return undefined;
    if (type !== 'position') setPending(true);
    setError('');
    try {
      const value = await request<Project>(`/projects/${current.current.project_id}/commands`, { type, revision: current.current.revision, ...values });
      if (current.current?.project_id === projectId) update(value); return value;
    } catch (e) {
      if (type === 'relink' && e instanceof ApiError && e.detail.code === 'source_mismatch') setRecovery({ videoPath: String(values.video_path), changed: true, relink: true });
      fail(e);
      // A concurrent job or another tab may have advanced the revision.
      try { const refreshed = await request<Project>(`/projects/${projectId}`); if (current.current?.project_id === projectId) update(refreshed); } catch { /* retain reviewable snapshot */ }
    } finally { if (type !== 'position') setPending(false); }
  });
  const command: Command = useCallback((type, values = {}) => current.current
    ? queue.current!(type, values, current.current.project_id) : Promise.resolve(undefined), []);
  const open = async (id: string) => {
    try { const value = await request<Project>(`/projects/${id}`); update(value); setScreen('workspace'); setJob(null); if (value.busy_job_id) setJob(await request<Job>(`/jobs/${value.busy_job_id}`)); }
    catch (e) { fail(e); }
  };
  const importProject = async (path = openPath) => {
    if (!path.trim()) return;
    try { const value = await request<Project>('/projects/open', { path }); update(value); setScreen('workspace'); setJob(null); setOpenPath(''); }
    catch (e) {
      if (e instanceof ApiError && ['missing_source', 'source_mismatch'].includes(String(e.detail.code))) {
        setRecovery({ projectPath: path, videoPath: String(e.detail.video_path ?? ''), changed: e.detail.code === 'source_mismatch', relink: false }); setError('');
      } else fail(e);
    }
  };
  const browse = async () => { try { const path = await pickFile('project'); if (path) await importProject(path); } catch (e) { fail(e); } };
  const home = async () => { setScreen('projects'); setJob(null); await refreshProjects(); };
  const busy = pending || !!project?.busy_job_id || !!job && ['queued', 'running'].includes(job.status);
  useEffect(() => {
    if (!job || !['queued', 'running'].includes(job.status)) return;
    let cancelled = false;
    const timer = window.setInterval(async () => {
      try {
        const value = await request<Job>(`/jobs/${job.id}`);
        if (cancelled) return;
        setJob(value);
        if (!['queued', 'running'].includes(value.status) && current.current) {
          update(await request<Project>(`/projects/${current.current.project_id}`));
          if (value.status === 'failed') setError(value.error ?? value.message);
        }
      } catch (e) { if (!cancelled) fail(e); }
    }, 1000);
    return () => { cancelled = true; clearInterval(timer); };
  }, [job?.id, job?.status, update, fail]);
  useEffect(() => {
    const keydown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'z' && screen === 'workspace' && !statsOpen) {
        if ((event.target as HTMLElement).matches('input,textarea,[contenteditable=true]')) return;
        event.preventDefault(); if (!busy) void command(event.shiftKey ? 'redo' : 'undo');
      }
    };
    window.addEventListener('keydown', keydown); return () => window.removeEventListener('keydown', keydown);
  }, [screen, statsOpen, busy, command]);
  const relink = async () => {
    setRecovery({ videoPath: '', changed: false, relink: true });
  };
  const recoverSource = async () => {
    if (!recovery?.videoPath.trim()) return;
    if (recovery.relink) { const value = await command('relink', { video_path: recovery.videoPath, allow_source_mismatch: recovery.changed }); if (value) setRecovery(null); return; }
    try { update(await request<Project>('/projects/open', { path: recovery.projectPath, video_path: recovery.videoPath, allow_source_mismatch: recovery.changed })); setScreen('workspace'); setJob(null); setRecovery(null); setError(''); }
    catch (e) { if (e instanceof ApiError && e.detail.code === 'source_mismatch') { setRecovery({ ...recovery, changed: true }); setError(''); } else fail(e); }
  };
  return <div className="app-shell">
    <header className="topbar">
      <button className="brand" onClick={() => void home()} aria-label="Roundnet projects"><span className="brand-mark"><i/><i/><i/></span>roundnet<span className="brand-caption">RALLY EDITOR</span></button>
      {screen === 'workspace' && project ? <><span className="header-divider"/><div className="project-heading"><strong>{project.title}</strong><span className={`saved ${pending ? 'saving' : ''}`}><Icon name="check" size={13}/>{pending ? 'Saving…' : project.save_status === 'saved' ? 'All changes saved' : project.save_status}</span></div><div className="header-actions"><button className="icon-button" title="Undo (⌘Z)" disabled={busy || !project.can_undo} onClick={() => void command('undo')}><Icon name="undo"/></button><button className="icon-button" title="Redo (⇧⌘Z)" disabled={busy || !project.can_redo} onClick={() => void command('redo')}><Icon name="redo"/></button><span className="header-divider"/><button className="button subtle" onClick={() => setStatsOpen(true)}><Icon name="stats"/>Statistics</button><button className="button primary small" disabled={busy || !project.match_settings.setup_complete} onClick={() => void command('stage', { stage: 'export' })}><Icon name="export"/>Export</button></div></> : <span className="local-badge"><span/>ON YOUR COMPUTER</span>}
    </header>
    {error && <div className="error-banner" role="alert"><span>{error}</span><button className="icon-button" onClick={() => setError('')} aria-label="Dismiss error"><Icon name="close"/></button></div>}
    {screen === 'projects' && <main className="dashboard">
      <div className="dashboard-title"><div><div className="eyebrow">YOUR WORKSPACE</div><h1>Good games deserve<br/><span>a great replay.</span></h1><p>From the first serve to the final point.<br/>Find your rallies, tell the story, keep the memories.</p></div><div className="court-art" aria-hidden="true"><div className="court-orbit"/><div className="court-net"/><i className="court-ball"/><span className="court-coordinate">40.0° N · GAME ON</span></div></div>
      <div className="section-heading"><div><h2>Your matches <span className="count-pill">{projects.length}</span></h2><p>Pick up right where you left off.</p></div><div className="row"><button className="button" onClick={() => void browse()}><Icon name="folder"/>Open project</button><button className="button primary" onClick={() => { setProject(null); current.current = null; setJob(null); setScreen('new'); }}><Icon name="plus"/>New match</button></div></div>
      {loading ? <div className="empty-state"><span className="spinner"/>Loading your matches…</div> : <div className="project-grid"><button className="new-project-card" onClick={() => { setProject(null); current.current = null; setJob(null); setScreen('new'); }}><span className="new-icon"><Icon name="plus" size={27}/></span><strong>Start a new match</strong><span>Your recording. Your next great cut.</span></button>{projects.map(p => <button key={p.project_id} className="project-card" onClick={() => void open(p.project_id)}><div className="project-thumbnail"><img src={thumbnailUrl(p.project_id)} alt="" loading="lazy"/><span className="stage-pill">{stages.find(([stage]) => stage === p.stage)?.[1] ?? p.stage}</span></div><div className="project-card-body"><strong>{p.title}</strong><div><span>{p.rally_count ?? 0} clips</span><span>{p.saved_utc ? new Date(p.saved_utc).toLocaleDateString() : 'Just created'}</span></div></div></button>)}</div>}
      <details className="import-details"><summary>Open a project by its local path</summary><form className="row" onSubmit={e => { e.preventDefault(); void importProject(); }}><input value={openPath} onChange={e => setOpenPath(e.target.value)} placeholder="/path/to/match.roundnet.json" aria-label="Project file path"/><button className="button" type="submit">Open project</button></form></details>
      <footer className="dashboard-footer"><Icon name="video"/>Your footage stays here. No uploads, no account, just the game.<button onClick={() => void refreshProjects()}>Refresh matches</button></footer>
    </main>}
    {screen === 'new' && <Setup project={null} command={command} update={value => { update(value); setScreen('workspace'); }} onError={fail} onBack={() => void home()}/>}
    {screen === 'workspace' && project && <>
      <nav className="workflow-nav" aria-label="Match workflow"><button className="workspace-back" onClick={() => void home()}><Icon name="grid"/>Projects</button><div>{stages.map(([stage, title], index) => <button className={project.stage === stage ? 'active' : ''} key={stage} disabled={busy || stage !== 'setup' && !project.match_settings.setup_complete} onClick={() => void command('stage', { stage })}><span>{index + 1}</span>{title}{index < stages.length - 1 && <Icon name="chevron" size={14} style={{ transform: 'rotate(-90deg)' }}/>}</button>)}</div><button className="icon-button" title="Project settings" disabled={busy} onClick={() => void command('stage', { stage: 'setup' })}><Icon name="settings"/></button></nav>
      {(project.source_missing || project.source_changed) && <div className="warning-banner">{project.source_missing ? 'The recording has moved. Relink it to resume editing.' : 'The recording changed since this project was saved. Relink the original recording before editing.'}<button className="button small" onClick={() => void relink()}>Relink recording</button></div>}
      <fieldset className="workspace-fieldset" disabled={!!project.source_missing || !!project.source_changed}>
        {project.stage === 'setup' && <Setup key={project.project_id} project={project} command={command} update={update} onError={fail} onBack={() => void home()}/>}
        {project.stage === 'find' && <FindClips project={project} command={command} job={job} onJob={setJob} onError={fail} busy={busy}/>}
        {(project.stage === 'review' || project.stage === 'statistics') && <Review project={project} command={command} busy={busy} shortcuts={!statsOpen && !recovery && savePath === null} onStats={() => setStatsOpen(true)}/>}
        {project.stage === 'export' && <Export project={project} command={command} job={job} onJob={setJob} onError={fail} busy={busy}/>}
      </fieldset>
    </>}
    {statsOpen && project && <Statistics project={project} command={command} onClose={() => setStatsOpen(false)}/>}
    {screen === 'workspace' && project && <button className="save-copy-button" disabled={busy} onClick={() => setSavePath(project.project_path ?? '')}><Icon name="folder" size={13}/>Save a project copy</button>}
    {savePath !== null && <Modal className="recovery-modal" labelledBy="save-title" onClose={() => setSavePath(null)}><div className="modal-heading"><h2 id="save-title">Save a project copy</h2><button className="icon-button" aria-label="Close save dialog" onClick={() => setSavePath(null)}><Icon name="close"/></button></div><p>Your source recording stays separate. This file can be reopened in either interface.</p><label>Project file path<input autoFocus value={savePath} onChange={e => setSavePath(e.target.value)} placeholder="/path/to/match.roundnet.json"/></label><div className="modal-actions"><button className="button" onClick={() => setSavePath(null)}>Cancel</button><button className="button primary" disabled={pending || !savePath.trim()} onClick={async () => { if (await command('save_as', { path: savePath })) setSavePath(null); }}>Save project</button></div></Modal>}
    {recovery && <Modal className="recovery-modal" labelledBy="recovery-title" onClose={() => setRecovery(null)}><div className="modal-heading"><h2 id="recovery-title">{recovery.changed ? 'Confirm a changed recording' : 'Locate the original recording'}</h2><button className="icon-button" aria-label="Close source recovery" onClick={() => setRecovery(null)}><Icon name="close"/></button></div><p>{recovery.changed ? 'This file differs from the saved recording. Using it keeps valid clip boundaries and clears old outcomes, touches, detection signals, and undo history. Review and classify the clips again.' : 'Choose the original video to resume your saved work. Your project file does not contain the recording.'}</p><label>Recording path<input autoFocus value={recovery.videoPath} onChange={e => setRecovery({ ...recovery, videoPath: e.target.value, changed: false })} placeholder="/path/to/original/game.mov"/></label><button className="text-button" onClick={async () => { try { const path = await pickFile('video'); if (path) setRecovery({ ...recovery, videoPath: path, changed: false }); } catch (e) { fail(e); } }}><Icon name="folder" size={15}/>Browse recordings</button><div className="modal-actions"><button className="button" onClick={() => setRecovery(null)}>Cancel</button><button className="button primary" disabled={pending || !recovery.videoPath.trim()} onClick={() => void recoverSource()}>{recovery.changed ? 'Use changed recording' : 'Resume match'}<Icon name="arrow"/></button></div></Modal>}
  </div>;
}
