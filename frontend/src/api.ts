export class ApiError extends Error {
  constructor(message: string, public detail: Record<string, unknown> = {}, public status = 0) { super(message); }
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
export const cardUrl = (id: string, revision: number, aspect = 'source') => `/api/projects/${id}/card?revision=${revision}&aspect_ratio=${encodeURIComponent(aspect)}`;
export async function pickFile(kind: 'video' | 'project' | 'destination'): Promise<string> {
  const result = await request<{ path?: string | null }>('/files/pick', { kind });
  return result.path ?? '';
}
type MediaSource = { project_id: string; video_path?: string; source_stat?: { size: number; mtime_ns: number } };
const mediaSources = new Map<string, string>();
export const mediaVersion = (source: MediaSource) => JSON.stringify([source.video_path ?? '', source.source_stat?.size ?? '', source.source_stat?.mtime_ns ?? '']);
export function registerMediaSource(source: MediaSource) { mediaSources.set(source.project_id, mediaVersion(source)); }
