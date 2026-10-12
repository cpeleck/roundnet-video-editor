export class ApiError extends Error {
  constructor(message: string, public detail: Record<string, unknown> = {}, public status = 0) { super(message); }
}
export const isPhoneConnection = () => !['localhost', '127.0.0.1', '[::1]'].includes(window.location.hostname);

export function uploadRecording(file: File, onProgress: (percent: number) => void, signal: AbortSignal): Promise<{ path: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `/api/recordings?filename=${encodeURIComponent(file.name)}`);
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    const abort = () => xhr.abort();
    const finish = () => signal.removeEventListener('abort', abort);
    xhr.upload.onprogress = event => { if (event.lengthComputable) onProgress(Math.round(event.loaded / event.total * 100)); };
    xhr.onload = () => {
      finish();
      try {
        const data = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) resolve(data);
        else reject(new Error(typeof data.detail === 'string' ? data.detail : data.detail?.message ?? 'The recording could not be transferred.'));
      } catch { reject(new Error('The recording could not be transferred. Check that your Mac is still running.')); }
    };
    xhr.onerror = () => { finish(); reject(new Error('Transfer interrupted. Keep Safari open and both devices on the same Wi-Fi, then try again.')); };
    xhr.onabort = () => { finish(); reject(new DOMException('Transfer cancelled', 'AbortError')); };
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) { finish(); reject(new DOMException('Transfer cancelled', 'AbortError')); }
    else xhr.send(file);
  });
}
export async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api${path}`, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`; let detail: Record<string, unknown> = {};
    try { const data = await response.json(); detail = typeof data.detail === 'object' ? data.detail : {}; message = typeof data.detail === 'string' ? data.detail : detail.message ? String(detail.message) : JSON.stringify(data.detail ?? data); } catch { /* non-JSON service failure */ }
    throw new ApiError(message, detail, response.status);
  }
  return response.json() as Promise<T>;
}
export const mediaUrl = (id: string, version = mediaSources.get(id) ?? '') => `/api/projects/${id}/media?v=${encodeURIComponent(version)}`;
export const thumbnailUrl = (id: string, time = 0, width = 400) => `/api/projects/${id}/thumbnail?time=${time}&width=${width}&v=${encodeURIComponent(mediaSources.get(id) ?? '')}`;
export const cardUrl = (id: string, revision: number, aspect = 'source', resolution = 'source') => `/api/projects/${id}/card?revision=${revision}&aspect_ratio=${encodeURIComponent(aspect)}&resolution=${encodeURIComponent(resolution)}`;
export async function pickFile(kind: 'video' | 'project' | 'destination'): Promise<string> {
  const result = await request<{ path?: string | null }>('/files/pick', { kind });
  return result.path ?? '';
}
type MediaSource = { project_id: string; video_path?: string; source_stat?: { size: number; mtime_ns: number } };
const mediaSources = new Map<string, string>();
export const mediaVersion = (source: MediaSource) => JSON.stringify([source.video_path ?? '', source.source_stat?.size ?? '', source.source_stat?.mtime_ns ?? '']);
export function registerMediaSource(source: MediaSource) { mediaSources.set(source.project_id, mediaVersion(source)); }
