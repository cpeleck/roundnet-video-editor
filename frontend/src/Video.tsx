import { useEffect, useState, type RefObject } from 'react';
import { mediaUrl, mediaVersion, request } from './api';
import type { Project } from './types';

export function Video({ project, videoRef, onTime, onLoaded, onPlay, onPause, onEnded, onClick }: { project: Project; videoRef: RefObject<HTMLVideoElement | null>; onTime?: () => void; onLoaded?: () => void; onPlay?: () => void; onPause?: () => void; onEnded?: () => void; onClick?: (event: React.MouseEvent<HTMLVideoElement>) => void }) {
  const [status, setStatus] = useState('loading');
  const [message, setMessage] = useState('Preparing your recording…');
  const version = mediaVersion(project);
  const sourceIdentity = `${project.project_id}:${version}`;
  const [playback, setPlayback] = useState({ sourceIdentity, proxy: false, retry: 0 });
  // Retry choices belong to one source. A relink begins fresh even when the
  // preceding recording failed playback or required a proxy.
  const proxy = playback.sourceIdentity === sourceIdentity && playback.proxy;
  const retry = playback.sourceIdentity === sourceIdentity ? playback.retry : 0;
  useEffect(() => {
    let disposed = false; let timer = 0;
    setStatus('loading');
    const poll = async () => {
      try {
        const query = new URLSearchParams({ v: version }); if (proxy) query.set('proxy', 'true'); if (retry && status === 'failed') query.set('retry', 'true');
        const info = await request<{ status: string; error?: string; job?: { progress: number; message?: string; error?: string } }>(`/projects/${project.project_id}/media-info?${query}`);
        if (disposed) return;
        setStatus(info.status); setMessage(info.error ?? info.job?.error ?? `Preparing playback preview${info.job ? ` · ${Math.round(info.job.progress)}%` : ''}…`);
        if (!['ready', 'failed', 'cancelled'].includes(info.status)) timer = window.setTimeout(poll, 1000);
      } catch (e) { if (!disposed) { setStatus('failed'); setMessage(e instanceof Error ? e.message : String(e)); } }
    };
    void poll(); return () => { disposed = true; clearTimeout(timer); };
  }, [project.project_id, version, proxy, retry]);
  return status === 'ready' ? <video key={sourceIdentity} ref={videoRef} src={`${mediaUrl(project.project_id, version)}${proxy ? '&proxy=true' : ''}`} preload="metadata" playsInline onLoadedMetadata={onLoaded} onTimeUpdate={onTime} onPlay={onPlay} onPause={onPause} onEnded={onEnded} onClick={onClick} onError={() => { if (!proxy) setPlayback({ sourceIdentity, proxy: true, retry }); else { setStatus('failed'); setMessage('This preview could not be played. Retry playback or check your FFmpeg installation.'); } }}/>
    : <div className="video-placeholder" role="status"><span className={status === 'failed' ? '' : 'spinner'}/><p>{message}</p>{['failed', 'cancelled'].includes(status) && <button className="button" onClick={() => setPlayback({ sourceIdentity, proxy: true, retry: retry + 1 })}>Retry playback</button>}</div>;
}
