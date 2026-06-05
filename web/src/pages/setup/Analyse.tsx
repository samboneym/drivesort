import { useEffect, useRef, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import Plot from 'react-plotly.js'
import { Loader2, CheckCircle2, AlertCircle } from 'lucide-react'

import { api } from '@/lib/api'
import { useWebSocket } from '@/lib/useWebSocket'
import type { AnalysisPhase, UmapPoint, WsEvent } from '@/lib/types'

// ---- Constants ----------------------------------------------------------------

const CLUSTER_PALETTE = [
  '#6366f1', '#06b6d4', '#10b981', '#f59e0b',
  '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6',
]

const ORDERED_PHASES: AnalysisPhase[] = ['fetching', 'embedding', 'clustering', 'naming']

const PHASE_LABELS: Record<string, string> = {
  fetching:   'Fetch',
  embedding:  'Embed',
  clustering: 'Cluster',
  naming:     'Name',
}

const PHASE_STATIC_DETAIL: Record<string, string> = {
  fetching:   'Discovering files in Drive',
  clustering: 'Computing UMAP layout and running HDBSCAN',
  naming:     'Asking Ollama to name each cluster',
}

// ---- Types --------------------------------------------------------------------

interface EmbedProgress { done: number; total: number; cached: number }

// ---- Phase row ----------------------------------------------------------------

function PhaseIcon({ status }: { status: 'pending' | 'active' | 'done' | 'error' }) {
  if (status === 'active')
    return <Loader2 size={16} className="text-indigo-500 animate-spin flex-shrink-0" />
  if (status === 'done')
    return <CheckCircle2 size={16} className="text-green-500 flex-shrink-0" />
  if (status === 'error')
    return <AlertCircle size={16} className="text-red-400 flex-shrink-0" />
  return (
    <span className="flex-shrink-0 w-4 h-4 rounded-full border border-gray-600" aria-hidden="true" />
  )
}

interface PhaseRowProps {
  phase: AnalysisPhase
  status: 'pending' | 'active' | 'done' | 'error'
  totalFiles: number
  embedProgress: EmbedProgress
}

function PhaseRow({ phase, status, totalFiles, embedProgress }: PhaseRowProps) {
  const label = PHASE_LABELS[phase] ?? phase

  let detail: React.ReactNode = null

  if (phase === 'fetching') {
    if (status === 'done' && totalFiles > 0)
      detail = <span>{totalFiles} files found</span>
    else if (status === 'active' || status === 'pending')
      detail = <span>{PHASE_STATIC_DETAIL.fetching}</span>
  } else if (phase === 'embedding') {
    if (status === 'active') {
      const pct = embedProgress.total > 0
        ? (embedProgress.done / embedProgress.total) * 100
        : 0
      detail = (
        <div className="space-y-1 w-full">
          <div className="w-full bg-gray-800 rounded-full h-1.5 overflow-hidden">
            <div
              className="h-1.5 bg-indigo-500 rounded-full transition-all duration-200"
              style={{ width: `${pct}%` }}
            />
          </div>
          <span>
            {embedProgress.done} / {embedProgress.total} embedded
            {embedProgress.cached > 0 && ` · ${embedProgress.cached} cached`}
          </span>
        </div>
      )
    } else if (status === 'done') {
      detail = (
        <span>
          {embedProgress.done} / {embedProgress.total} embedded
          {embedProgress.cached > 0 && ` · ${embedProgress.cached} cached`}
        </span>
      )
    }
  } else if (phase === 'clustering') {
    detail = <span>{PHASE_STATIC_DETAIL.clustering}</span>
  } else if (phase === 'naming') {
    detail = <span>{PHASE_STATIC_DETAIL.naming}</span>
  }

  return (
    <div className="flex items-start gap-3 py-2">
      <div className="mt-0.5">
        <PhaseIcon status={status} />
      </div>
      <div className="flex flex-col gap-1 min-w-0 flex-1">
        <span className="text-white text-sm font-medium">{label}</span>
        {detail && (
          <span className="text-gray-400 text-sm">{detail}</span>
        )}
      </div>
    </div>
  )
}

// ---- Scatter plot -------------------------------------------------------------

function UmapScatter({ points }: { points: UmapPoint[] }) {
  const noisePoints = points.filter(p => p.label === -1)
  const clusterPoints = points.filter(p => p.label >= 0)

  // Group cluster points by label for per-cluster colour
  const byLabel = new Map<number, UmapPoint[]>()
  for (const p of clusterPoints) {
    if (!byLabel.has(p.label)) byLabel.set(p.label, [])
    byLabel.get(p.label)!.push(p)
  }

  const traces: Plotly.Data[] = []

  // Noise trace
  if (noisePoints.length > 0) {
    traces.push({
      type: 'scatter',
      mode: 'markers',
      x: noisePoints.map(p => p.x),
      y: noisePoints.map(p => p.y),
      marker: { color: '#4b5563', size: 3 },
      hoverinfo: 'none',
      showlegend: false,
    })
  }

  // Per-cluster traces
  for (const [label, pts] of byLabel.entries()) {
    const color = CLUSTER_PALETTE[label % CLUSTER_PALETTE.length]
    traces.push({
      type: 'scatter',
      mode: 'markers',
      x: pts.map(p => p.x),
      y: pts.map(p => p.y),
      marker: { color, size: 5 },
      hoverinfo: 'none',
      showlegend: false,
    })
  }

  return (
    <Plot
      data={traces}
      layout={{
        paper_bgcolor: '#0f1117',
        plot_bgcolor: '#0f1117',
        height: 320,
        margin: { t: 8, b: 8, l: 8, r: 8 },
        xaxis: {
          showgrid: false,
          zeroline: false,
          showticklabels: false,
          showline: false,
        },
        yaxis: {
          showgrid: false,
          zeroline: false,
          showticklabels: false,
          showline: false,
        },
      }}
      config={{ displayModeBar: false, staticPlot: true }}
      style={{ width: '100%' }}
      useResizeHandler
    />
  )
}

// ---- Main component -----------------------------------------------------------

export function Analyse() {
  const navigate = useNavigate()

  const [phase, setPhase]             = useState<AnalysisPhase>('idle')
  const [totalFiles, setTotalFiles]   = useState(0)
  const [embedProgress, setEmbed]     = useState<EmbedProgress>({ done: 0, total: 0, cached: 0 })
  const [umapPoints, setUmapPoints]   = useState<UmapPoint[]>([])
  const [cacheHit, setCacheHit]       = useState(false)
  const [errorMsg, setErrorMsg]       = useState<string | null>(null)

  // Track which phases are "done" so they stay green when later phases activate
  const [donePhases, setDonePhases]   = useState<Set<AnalysisPhase>>(new Set())

  // Batch UMAP point updates every 200 ms
  const pendingPointsRef = useRef<UmapPoint[]>([])
  const flushTimerRef    = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    flushTimerRef.current = setInterval(() => {
      if (pendingPointsRef.current.length === 0) return
      const batch = pendingPointsRef.current.splice(0)
      setUmapPoints(prev => [...prev, ...batch])
    }, 200)
    return () => {
      if (flushTimerRef.current) clearInterval(flushTimerRef.current)
    }
  }, [])

  const markPhaseDone = useCallback((p: AnalysisPhase) => {
    setDonePhases(prev => new Set([...prev, p]))
  }, [])

  const handleEvent = useCallback((event: WsEvent) => {
    switch (event.type) {
      case 'phase': {
        const prev = phase
        // Mark the previous phase done when a new phase begins
        if (prev !== 'idle' && prev !== 'complete' && prev !== 'error') {
          markPhaseDone(prev)
        }
        setPhase(event.phase)
        if (event.phase === 'complete') {
          // Mark all ordered phases done
          setDonePhases(new Set(ORDERED_PHASES))
          setTimeout(() => navigate('/setup/review'), 1500)
        }
        break
      }
      case 'fetch_complete':
        setTotalFiles(event.total)
        markPhaseDone('fetching')
        break
      case 'embed_progress':
        setEmbed({ done: event.done, total: event.total, cached: event.cached })
        break
      case 'umap_point':
        pendingPointsRef.current.push({
          file_id: event.file_id,
          x: event.x,
          y: event.y,
          label: event.label,
        })
        break
      case 'cluster_cache_hit':
        setCacheHit(true)
        setPhase('complete')
        setDonePhases(new Set(ORDERED_PHASES))
        setTimeout(() => navigate('/setup/review'), 1500)
        break
      case 'error':
        setPhase('error')
        setErrorMsg(event.message)
        break
    }
  }, [phase, markPhaseDone, navigate])

  useWebSocket(handleEvent)

  // On mount: check status, start if not already complete
  useEffect(() => {
    api.analysis.status()
      .then(status => {
        if (status.phase === 'complete') {
          navigate('/setup/review', { replace: true })
        } else {
          api.analysis.trigger().catch(() => {})
        }
      })
      .catch(() => {
        api.analysis.trigger().catch(() => {})
      })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleRetry = () => {
    setPhase('idle')
    setErrorMsg(null)
    setDonePhases(new Set())
    setUmapPoints([])
    setTotalFiles(0)
    setEmbed({ done: 0, total: 0, cached: 0 })
    api.analysis.trigger().catch(() => {})
  }

  // Determine per-row status
  function rowStatus(p: AnalysisPhase): 'pending' | 'active' | 'done' | 'error' {
    if (phase === 'error' && p === phase) return 'error'
    if (donePhases.has(p)) return 'done'
    if (phase === p) return 'active'
    return 'pending'
  }

  // ---- Error state ------------------------------------------------------------
  if (phase === 'error' && errorMsg) {
    return (
      <div className="h-full overflow-y-auto">
        <div className="max-w-2xl mx-auto px-6 py-16">
          <div className="border border-red-500/40 bg-red-500/5 rounded-xl p-6 space-y-4">
            <div className="flex items-center gap-2">
              <AlertCircle size={18} className="text-red-400 flex-shrink-0" />
              <span className="text-red-400 text-sm font-medium">Analysis failed</span>
            </div>
            <p className="text-gray-300 text-sm">{errorMsg}</p>
            <button
              onClick={handleRetry}
              className="mt-2 px-4 py-2 rounded-lg bg-indigo-500 hover:bg-indigo-600 text-white text-sm font-medium transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    )
  }

  // ---- Cache hit banner -------------------------------------------------------
  const showCacheHitBanner = cacheHit && phase === 'complete'

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-2xl mx-auto px-6 py-16">

        {/* Header */}
        <div>
          <h1 className="text-white text-2xl font-semibold">Analysing your Drive</h1>
          <p className="text-gray-400 text-sm mt-1">
            Building your file taxonomy — this takes a few minutes on first run
          </p>
        </div>

        {/* Cache hit banner */}
        {showCacheHitBanner && (
          <div className="mt-10 bg-indigo-500/10 border border-indigo-500/30 rounded-xl p-4">
            <p className="text-indigo-300 text-sm">
              &#x26A1; Loaded from cache — 0 files changed since last analysis
            </p>
          </div>
        )}

        {/* Phase progress list */}
        {!showCacheHitBanner && (
          <div className="mt-10 divide-y divide-gray-800">
            {ORDERED_PHASES.map(p => (
              <PhaseRow
                key={p}
                phase={p}
                status={rowStatus(p)}
                totalFiles={totalFiles}
                embedProgress={embedProgress}
              />
            ))}
          </div>
        )}

        {/* UMAP scatter plot */}
        {umapPoints.length > 0 && (
          <div className="mt-10">
            <UmapScatter points={umapPoints} />
          </div>
        )}

      </div>
    </div>
  )
}
