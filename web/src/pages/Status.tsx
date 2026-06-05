import { useCallback, useEffect, useState } from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  Database,
  Layers,
  Loader2,
  RefreshCw,
  Sparkles,
  Trash2,
  Zap,
} from 'lucide-react'

import { api } from '@/lib/api'
import type { CacheLayerStatus, CacheStatus, ScanStats } from '@/lib/types'

// ---- Helpers ------------------------------------------------------------------

function fmtBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`
}

function fmtDate(iso: string | null): string {
  if (!iso) return 'Never'
  return new Date(iso).toLocaleDateString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}

// ---- Stat card ----------------------------------------------------------------

interface StatCardProps {
  label: string
  value: number
  icon: React.ReactNode
  iconClass: string
}

function StatCard({ label, value, icon, iconClass }: StatCardProps) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex items-start gap-3">
      <div className={`mt-0.5 flex-shrink-0 ${iconClass}`}>{icon}</div>
      <div>
        <p className="text-white text-xl font-semibold leading-tight">{value}</p>
        <p className="text-gray-500 text-xs mt-0.5">{label}</p>
      </div>
    </div>
  )
}

// ---- Cache layer card ---------------------------------------------------------

interface CacheCardProps {
  title: string
  description: string
  icon: React.ReactNode
  status: CacheLayerStatus
  clearing: boolean
  confirmClear: boolean
  onClear: () => void
}

function CacheCard({ title, description, icon, status, clearing, confirmClear, onClear }: CacheCardProps) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex flex-col gap-3">
      <div className="flex items-start gap-2">
        <div className="text-gray-400 mt-0.5 flex-shrink-0">{icon}</div>
        <div className="flex-1 min-w-0">
          <p className="text-white text-sm font-medium">{title}</p>
          <p className="text-gray-500 text-xs mt-0.5 leading-snug">{description}</p>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 text-center">
        <div>
          <p className="text-white text-sm font-medium">{status.entries}</p>
          <p className="text-gray-600 text-xs">entries</p>
        </div>
        <div>
          <p className="text-white text-sm font-medium">{fmtBytes(status.size_bytes)}</p>
          <p className="text-gray-600 text-xs">on disk</p>
        </div>
        <div>
          <p className="text-gray-300 text-xs font-medium leading-tight">{fmtDate(status.last_updated)}</p>
          <p className="text-gray-600 text-xs">updated</p>
        </div>
      </div>

      <button
        onClick={onClear}
        disabled={clearing || status.entries === 0}
        className={[
          'flex items-center justify-center gap-1.5 h-8 w-full rounded-lg text-xs font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed',
          confirmClear
            ? 'bg-red-500/20 border border-red-500/40 text-red-400 hover:bg-red-500/30'
            : 'bg-gray-800 border border-gray-700 text-gray-400 hover:text-gray-200 hover:border-gray-600',
        ].join(' ')}
      >
        {clearing ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={12} />}
        {confirmClear ? 'Tap again to confirm' : 'Clear'}
      </button>
    </div>
  )
}

// ---- Status (main) -----------------------------------------------------------

const CACHE_META: Record<keyof CacheStatus, { title: string; description: string; icon: React.ReactNode }> = {
  content:    { title: 'Content',    description: 'Extracted text per file (Drive export, PDF, code)',   icon: <Database size={16} /> },
  embeddings: { title: 'Embeddings', description: 'Sentence-transformer vectors (all-MiniLM-L6-v2)',     icon: <Layers size={16} /> },
  clustering: { title: 'Clustering', description: 'UMAP layout + HDBSCAN cluster assignments snapshot',  icon: <Zap size={16} /> },
  llm_names:  { title: 'LLM Names',  description: 'Ollama-generated cluster labels',                     icon: <Sparkles size={16} /> },
}

type ClearKey = keyof CacheStatus | 'all'

export function Status() {
  const [stats, setStats]     = useState<ScanStats | null>(null)
  const [cache, setCache]     = useState<CacheStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState<string | null>(null)
  const [triggering, setTriggering]   = useState(false)
  const [clearing, setClearing]       = useState<Partial<Record<ClearKey, boolean>>>({})
  const [confirmClear, setConfirmClear] = useState<Partial<Record<ClearKey, boolean>>>({})

  const load = useCallback(() => {
    return Promise.all([api.scan.stats(), api.cache.status()])
      .then(([s, c]) => { setStats(s); setCache(c) })
      .catch(() => setError('Failed to load status.'))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => { load() }, [load])

  async function handleTrigger() {
    setTriggering(true)
    try { await api.scan.trigger() }
    catch { setError('Failed to trigger scan.') }
    finally { setTriggering(false) }
  }

  async function clearLayer(key: keyof CacheStatus) {
    if (!confirmClear[key]) {
      setConfirmClear(p => ({ ...p, [key]: true }))
      return
    }
    setClearing(p => ({ ...p, [key]: true }))
    setConfirmClear(p => ({ ...p, [key]: false }))
    try {
      await api.cache.clearLayer(key)
      await load()
    } catch {
      setError(`Failed to clear ${key} cache.`)
    } finally {
      setClearing(p => ({ ...p, [key]: false }))
    }
  }

  async function clearAll() {
    if (!confirmClear.all) {
      setConfirmClear(p => ({ ...p, all: true }))
      return
    }
    setClearing(p => ({ ...p, all: true }))
    setConfirmClear(p => ({ ...p, all: false }))
    try {
      await api.cache.clearAll()
      await load()
    } catch {
      setError('Failed to clear caches.')
    } finally {
      setClearing(p => ({ ...p, all: false }))
    }
  }

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 size={24} className="animate-spin text-gray-500" />
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-2xl mx-auto px-6 py-8 flex flex-col gap-8">

        {error && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-3 text-red-400 text-sm flex items-center gap-2">
            <AlertTriangle size={14} className="flex-shrink-0" />
            {error}
          </div>
        )}

        {/* ---- Scan trigger ---- */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-white font-medium">Scan Drive</h2>
            <p className="text-gray-500 text-xs mt-0.5">Discover and classify new files in the background</p>
          </div>
          <button
            onClick={handleTrigger}
            disabled={triggering}
            className="flex items-center gap-2 h-9 px-4 rounded-lg bg-indigo-500 hover:bg-indigo-600 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium transition-colors"
          >
            {triggering ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
            {triggering ? 'Scanning…' : 'Scan now'}
          </button>
        </div>

        {/* ---- Scan stats ---- */}
        {stats && (
          <div>
            <h3 className="text-gray-400 text-xs font-medium uppercase tracking-wide mb-3">Review Stats</h3>
            <div className="grid grid-cols-3 gap-3">
              <StatCard label="In queue"  value={stats.queued}    icon={<Database size={16} />}    iconClass="text-amber-400" />
              <StatCard label="Accepted"  value={stats.accepted}  icon={<CheckCircle2 size={16} />} iconClass="text-green-500" />
              <StatCard label="Corrected" value={stats.corrected} icon={<Zap size={16} />}          iconClass="text-indigo-400" />
            </div>
          </div>
        )}

        {/* ---- Cache cards ---- */}
        {cache && (
          <div>
            <h3 className="text-gray-400 text-xs font-medium uppercase tracking-wide mb-3">Cache Layers</h3>
            <div className="grid grid-cols-2 gap-3">
              {(Object.keys(CACHE_META) as (keyof CacheStatus)[]).map(key => (
                <CacheCard
                  key={key}
                  {...CACHE_META[key]}
                  status={cache[key]}
                  clearing={!!clearing[key]}
                  confirmClear={!!confirmClear[key]}
                  onClear={() => clearLayer(key)}
                />
              ))}
            </div>

            <button
              onClick={clearAll}
              disabled={!!clearing.all}
              className={[
                'mt-4 flex items-center justify-center gap-2 h-10 w-full rounded-xl text-sm font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed',
                confirmClear.all
                  ? 'bg-red-500/20 border border-red-500/40 text-red-400 hover:bg-red-500/30'
                  : 'bg-gray-900 border border-gray-800 text-gray-500 hover:text-gray-300 hover:border-gray-700',
              ].join(' ')}
            >
              {clearing.all ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />}
              {confirmClear.all ? 'Tap again — this cannot be undone' : 'Clear all caches'}
            </button>
          </div>
        )}

      </div>
    </div>
  )
}
