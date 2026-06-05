import type {
  AuthStatus, AnalysisStatus, AnalysisResult,
  TaxonomyNode, AddNodePayload, PatchNodePayload, ConfirmPayload,
  DraftState, SaveDraftPayload,
  StageEntry,
  ScanQueueItem, CorrectPayload,
  CacheStatus,
  DriveFileMeta,
} from './types'

async function get<T>(url: string): Promise<T> {
  const r = await fetch(url)
  return r.json() as Promise<T>
}

async function post<T>(url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method: 'POST',
    ...(body !== undefined
      ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
      : {}),
  })
  return r.json() as Promise<T>
}

async function put<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return r.json() as Promise<T>
}

async function patch<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return r.json() as Promise<T>
}

async function del<T>(url: string): Promise<T> {
  const r = await fetch(url, { method: 'DELETE' })
  return r.json() as Promise<T>
}

function encodePath(p: string) {
  return encodeURIComponent(p)
}

export const api = {
  auth: {
    status: () => get<AuthStatus>('/api/auth/status'),
    login:  () => get<{ auth_url: string }>('/api/auth/login'),
  },

  analysis: {
    trigger: () => post<{ status: string }>('/api/analysis/trigger'),
    status:  () => get<AnalysisStatus>('/api/analysis/status'),
    result:  () => get<AnalysisResult>('/api/analysis/result'),
  },

  taxonomy: {
    list:    ()                           => get<TaxonomyNode[]>('/api/taxonomy/nodes'),
    add:     (p: AddNodePayload)          => post<TaxonomyNode>('/api/taxonomy/nodes', p),
    patch:   (path: string, p: PatchNodePayload) => patch<TaxonomyNode>(`/api/taxonomy/nodes/${encodePath(path)}`, p),
    delete:  (path: string)              => del<{ deleted: string }>(`/api/taxonomy/nodes/${encodePath(path)}`),
    confirm: (path: string, p: ConfirmPayload) => post<{ ok: boolean }>(`/api/taxonomy/nodes/${encodePath(path)}/confirm`, p),
  },

  draft: {
    get:     () => get<DraftState | null>('/api/draft'),
    save:    (p: SaveDraftPayload) => put<{ saved: boolean }>('/api/draft', p),
    discard: () => del<{ discarded: boolean }>('/api/draft'),
  },

  stage: {
    list:   () => get<StageEntry[]>('/api/stage'),
    commit: () => post<{ committed: number }>('/api/stage/commit'),
  },

  scan: {
    trigger: () => post<{ status: string }>('/api/scan/trigger'),
    queue:   () => get<ScanQueueItem[]>('/api/scan/queue'),
    accept:  (fileId: string) => post<{ accepted: string; path: string | null }>(`/api/scan/queue/${fileId}/accept`),
    correct: (fileId: string, p: CorrectPayload) => post<{ corrected: string; path: string }>(`/api/scan/queue/${fileId}/correct`, p),
  },

  files: {
    list: () => get<DriveFileMeta[]>('/api/files'),
  },

  cache: {
    status:          () => get<CacheStatus>('/api/cache/status'),
    invalidateFile:  (fileId: string)   => post<{ invalidated: boolean }>('/api/cache/invalidate/file', { file_id: fileId }),
    invalidateFolder:(folderId: string) => post<{ invalidated: number }>('/api/cache/invalidate/folder', { folder_id: folderId }),
    clearAll:        () => del<{ cleared: boolean }>('/api/cache/all'),
  },
}
