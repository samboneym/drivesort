// ---- Auth ----

export interface AuthStatus {
  authenticated: boolean
  email: string | null
  auth_url?: string
}

// ---- Analysis ----

export type AnalysisPhase =
  | 'idle' | 'fetching' | 'embedding'
  | 'clustering' | 'naming' | 'complete' | 'error'

export interface AnalysisStatus {
  phase: AnalysisPhase
  progress: number
  total: number
  message: string
  error: string | null
}

export interface AnalysisResult {
  file_count: number
  cluster_count: number
}

// ---- Taxonomy ----

export interface TaxonomyNode {
  path: string
  name: string
  parent: string | null
  centroid: number[]
  member_ids: string[]
  member_count: number
  folder_id: string
  description: string
}

export interface AddNodePayload {
  path: string
  name: string
  parent: string | null
  member_embeddings: number[][]
  member_ids: string[]
  folder_id?: string
  description?: string
}

export interface PatchNodePayload {
  name?: string
  description?: string
}

export interface ConfirmPayload {
  embedding: number[]
  file_id: string
}

// ---- Draft ----

export interface StagedChange {
  file_id: string
  file_name: string
  current_path: string | null
  proposed_path: string | null
}

export interface UserDecision {
  file_id: string
  action: 'assign' | 'skip'
  path: string | null
  timestamp: string
}

export interface DraftState {
  saved_at: string
  taxonomy_tree: Record<string, unknown>
  staged_changes: StagedChange[]
  user_decisions: UserDecision[]
}

export interface SaveDraftPayload {
  taxonomy_tree: Record<string, unknown>
  staged_changes: StagedChange[]
  user_decisions: UserDecision[]
}

// ---- Stage ----

export interface StageEntry {
  file_id: string
  file_name: string
  source_folder_id: string | null
  target_folder_id: string
  target_path: string
}

// ---- Scan ----

export interface ScanQueueItem {
  file_id: string
  file_name: string
  mime_type: string
  predicted_path: string | null
  confidence: number
  is_novel: boolean
  embedding: number[]
}

export interface CorrectPayload {
  path: string
  embedding: number[]
}

// ---- Cache ----

export interface CacheLayerStatus {
  entries: number
  size_bytes: number
  last_updated: string | null
}

export interface CacheStatus {
  content: CacheLayerStatus
  embeddings: CacheLayerStatus
  clustering: CacheLayerStatus
  llm_names: CacheLayerStatus
}

// ---- WebSocket Events ----

export type WsEvent =
  | { type: 'phase'; phase: AnalysisPhase }
  | { type: 'fetch_complete'; total: number }
  | { type: 'embed_progress'; done: number; total: number; cached: number }
  | { type: 'umap_point'; file_id: string; x: number; y: number; label: number }
  | { type: 'cluster_cache_hit' }
  | { type: 'scan_file'; file_id: string; file_name: string; mime_type: string; predicted_path: string | null; confidence: number; is_novel: boolean; embedding: number[] }
  | { type: 'error'; message: string }

// ---- UMAP ----

export interface UmapPoint {
  file_id: string
  x: number
  y: number
  label: number
}
