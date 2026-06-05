import { useEffect, useRef, useState } from 'react'
import {
  CheckCircle2,
  ChevronRight,
  File,
  FileText,
  Film,
  Image,
  Loader2,
  Music,
  RefreshCw,
  Sparkles,
  Table,
  X,
} from 'lucide-react'

import { api } from '@/lib/api'
import { useWebSocket } from '@/lib/useWebSocket'
import type { ScanQueueItem, TaxonomyNode, WsEvent } from '@/lib/types'

// ---- MIME type icon -----------------------------------------------------------

function MimeIcon({ mimeType }: { mimeType: string }) {
  const cls = 'flex-shrink-0'
  if (mimeType.startsWith('image/'))                            return <Image size={14} className={cls} />
  if (mimeType.startsWith('video/'))                            return <Film size={14} className={cls} />
  if (mimeType.startsWith('audio/'))                            return <Music size={14} className={cls} />
  if (mimeType === 'application/pdf')                           return <FileText size={14} className={cls} />
  if (mimeType.includes('spreadsheet') || mimeType.includes('excel'))
                                                                return <Table size={14} className={cls} />
  if (mimeType.includes('document') || mimeType.includes('word'))
                                                                return <FileText size={14} className={cls} />
  return <File size={14} className={cls} />
}

// ---- Confidence bar -----------------------------------------------------------

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  const color = value >= 0.82
    ? 'bg-green-500'
    : value >= 0.62
    ? 'bg-amber-400'
    : 'bg-red-500'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-gray-400 w-8 text-right">{pct}%</span>
    </div>
  )
}

// ---- Queue item row -----------------------------------------------------------

interface QueueRowProps {
  item: ScanQueueItem
  active: boolean
  onClick: () => void
}

function QueueRow({ item, active, onClick }: QueueRowProps) {
  return (
    <button
      onClick={onClick}
      className={[
        'w-full text-left flex items-start gap-2 px-3 py-2.5 border-b border-gray-800/50 hover:bg-gray-800/40 transition-colors',
        active ? 'bg-gray-800/60' : '',
      ].join(' ')}
    >
      <div className="mt-0.5 text-gray-500">
        <MimeIcon mimeType={item.mime_type} />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-gray-200 truncate leading-tight">{item.file_name}</p>
        {item.predicted_path && (
          <p className="text-xs text-gray-500 truncate mt-0.5">{item.predicted_path}</p>
        )}
      </div>
      {item.is_novel && (
        <span className="flex-shrink-0 mt-0.5 text-xs bg-violet-500/15 text-violet-400 rounded px-1.5 py-0.5">
          new
        </span>
      )}
    </button>
  )
}

// ---- Empty states -------------------------------------------------------------

function EmptyQueue() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center p-6">
      <CheckCircle2 size={32} className="text-green-500" />
      <p className="text-white text-sm font-medium">All caught up</p>
      <p className="text-gray-500 text-xs">No files awaiting review.</p>
    </div>
  )
}

function NoSelection() {
  return (
    <div className="flex-1 flex items-center justify-center text-gray-600 text-sm">
      Select a file from the queue
    </div>
  )
}

// ---- Taxonomy path list (scrollable) -----------------------------------------

interface PathListProps {
  nodes: TaxonomyNode[]
  highlighted: string | null
}

function PathList({ nodes, highlighted }: PathListProps) {
  const sorted = [...nodes].sort((a, b) => a.path.localeCompare(b.path))
  const activeRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: 'nearest' })
  }, [highlighted])

  return (
    <div className="flex-1 overflow-y-auto min-h-0">
      {sorted.map(n => {
        const depth = n.path.split('/').length - 1
        const isActive = n.path === highlighted
        return (
          <div
            key={n.path}
            ref={isActive ? activeRef : undefined}
            className={[
              'flex items-center gap-1.5 h-8 text-sm transition-colors',
              isActive
                ? 'bg-indigo-500/20 text-indigo-300'
                : 'text-gray-400',
            ].join(' ')}
            style={{ paddingLeft: `${12 + depth * 14}px`, paddingRight: '12px' }}
          >
            <ChevronRight size={12} className="flex-shrink-0 text-gray-600" />
            <span className="truncate">{n.name}</span>
            {isActive && <span className="ml-auto text-xs text-indigo-400">← here</span>}
          </div>
        )
      })}
    </div>
  )
}

// ---- Scan (main) --------------------------------------------------------------

export function Scan() {
  const [queue, setQueue]               = useState<ScanQueueItem[]>([])
  const [activeId, setActiveId]         = useState<string | null>(null)
  const [nodes, setNodes]               = useState<TaxonomyNode[]>([])
  const [loading, setLoading]           = useState(true)
  const [acting, setActing]             = useState(false)
  const [correcting, setCorrecting]     = useState(false)
  const [correctionPath, setCorrectionPath] = useState<string>('')
  const [triggering, setTriggering]     = useState(false)
  const [error, setError]               = useState<string | null>(null)

  // ---- Load on mount ----------------------------------------------------------
  useEffect(() => {
    Promise.all([api.scan.queue(), api.taxonomy.list()])
      .then(([q, tax]) => {
        setQueue(q)
        setNodes(tax)
        if (q.length > 0) setActiveId(q[0].file_id)
      })
      .catch(() => setError('Failed to load scan data.'))
      .finally(() => setLoading(false))
  }, [])

  // ---- WebSocket: append new scan_file events ---------------------------------
  useWebSocket((e: WsEvent) => {
    if (e.type !== 'scan_file') return
    const item: ScanQueueItem = {
      file_id:        e.file_id,
      file_name:      e.file_name,
      mime_type:      e.mime_type,
      predicted_path: e.predicted_path,
      confidence:     e.confidence,
      is_novel:       e.is_novel,
      embedding:      e.embedding,
    }
    setQueue(prev => {
      if (prev.some(x => x.file_id === item.file_id)) return prev
      const next = [...prev, item]
      setActiveId(id => id ?? item.file_id)
      return next
    })
  })

  // ---- Derived ----------------------------------------------------------------
  const activeItem = queue.find(i => i.file_id === activeId) ?? null
  const sortedPaths = [...nodes]
    .sort((a, b) => a.path.localeCompare(b.path))
    .map(n => n.path)

  function removeFromQueue(fileId: string) {
    setQueue(prev => {
      const next = prev.filter(i => i.file_id !== fileId)
      setActiveId(curr => {
        if (curr !== fileId) return curr
        const idx = prev.findIndex(i => i.file_id === fileId)
        return next[idx] ? next[idx].file_id : next[next.length - 1]?.file_id ?? null
      })
      return next
    })
  }

  // ---- Actions ----------------------------------------------------------------
  async function handleAccept() {
    if (!activeItem) return
    setActing(true)
    setError(null)
    try {
      await api.scan.accept(activeItem.file_id)
      removeFromQueue(activeItem.file_id)
    } catch {
      setError('Accept failed. Please try again.')
    } finally {
      setActing(false)
    }
  }

  async function handleCorrect() {
    if (!activeItem || !correctionPath) return
    setActing(true)
    setError(null)
    try {
      await api.scan.correct(activeItem.file_id, {
        path: correctionPath,
        embedding: activeItem.embedding,
      })
      removeFromQueue(activeItem.file_id)
      setCorrecting(false)
      setCorrectionPath('')
    } catch {
      setError('Correction failed. Please try again.')
    } finally {
      setActing(false)
    }
  }

  function handleSkip() {
    if (!activeItem) return
    removeFromQueue(activeItem.file_id)
  }

  async function handleTrigger() {
    setTriggering(true)
    setError(null)
    try {
      await api.scan.trigger()
    } catch {
      setError('Failed to start scan.')
    } finally {
      setTriggering(false)
    }
  }

  // ---- Loading ----------------------------------------------------------------
  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 size={24} className="animate-spin text-gray-500" />
      </div>
    )
  }

  // ---- Render -----------------------------------------------------------------
  return (
    <div className="h-full flex overflow-hidden">

      {/* ---- LEFT: Queue (25%) ---- */}
      <div className="w-[25%] flex flex-col border-r border-gray-800 overflow-hidden">
        <div className="h-12 flex items-center justify-between px-4 border-b border-gray-800 flex-shrink-0">
          <span className="text-gray-400 text-xs font-medium uppercase tracking-wide">
            Queue
            {queue.length > 0 && (
              <span className="ml-1.5 text-gray-500">({queue.length})</span>
            )}
          </span>
          <button
            onClick={handleTrigger}
            disabled={triggering}
            title="Trigger scan"
            className="text-gray-500 hover:text-gray-300 disabled:opacity-40 transition-colors"
          >
            <RefreshCw size={14} className={triggering ? 'animate-spin' : ''} />
          </button>
        </div>

        {queue.length === 0 ? (
          <EmptyQueue />
        ) : (
          <div className="flex-1 overflow-y-auto">
            {queue.map(item => (
              <QueueRow
                key={item.file_id}
                item={item}
                active={item.file_id === activeId}
                onClick={() => { setActiveId(item.file_id); setCorrecting(false); setCorrectionPath('') }}
              />
            ))}
          </div>
        )}
      </div>

      {/* ---- CENTRE: File detail (50%) ---- */}
      <div className="w-[50%] flex flex-col border-r border-gray-800 overflow-hidden">
        <div className="h-12 flex items-center px-4 border-b border-gray-800 flex-shrink-0">
          <span className="text-gray-400 text-xs font-medium uppercase tracking-wide">Placement</span>
        </div>

        {!activeItem ? (
          <NoSelection />
        ) : (
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* File header */}
            <div className="px-5 pt-5 pb-4 border-b border-gray-800 flex-shrink-0">
              <div className="flex items-start gap-3">
                <div className="mt-0.5 text-gray-400">
                  <MimeIcon mimeType={activeItem.mime_type} />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-white font-medium text-sm leading-snug break-all">
                    {activeItem.file_name}
                  </p>
                  <p className="text-gray-500 text-xs mt-0.5">{activeItem.mime_type}</p>
                </div>
                {activeItem.is_novel && (
                  <div className="flex items-center gap-1 text-violet-400 text-xs bg-violet-500/10 rounded-full px-2 py-0.5 flex-shrink-0">
                    <Sparkles size={10} />
                    Novel
                  </div>
                )}
              </div>

              {/* Predicted path */}
              <div className="mt-4">
                <p className="text-xs text-gray-500 mb-1.5">Predicted folder</p>
                {activeItem.predicted_path ? (
                  <div className="bg-indigo-500/10 border border-indigo-500/20 rounded-lg px-3 py-2">
                    <p className="text-indigo-300 text-sm font-mono break-all">
                      {activeItem.predicted_path}
                    </p>
                  </div>
                ) : (
                  <div className="bg-gray-800/50 border border-gray-700 rounded-lg px-3 py-2">
                    <p className="text-gray-500 text-sm italic">No prediction — novel file</p>
                  </div>
                )}
              </div>

              {/* Confidence */}
              <div className="mt-3">
                <p className="text-xs text-gray-500 mb-1.5">Confidence</p>
                <ConfidenceBar value={activeItem.confidence} />
              </div>
            </div>

            {/* Taxonomy path list */}
            <div className="px-4 pt-3 pb-1 flex-shrink-0">
              <p className="text-xs text-gray-500 uppercase tracking-wide">Taxonomy</p>
            </div>
            <PathList nodes={nodes} highlighted={activeItem.predicted_path} />
          </div>
        )}
      </div>

      {/* ---- RIGHT: Actions (25%) ---- */}
      <div className="w-[25%] flex flex-col overflow-hidden">
        <div className="h-12 flex items-center px-4 border-b border-gray-800 flex-shrink-0">
          <span className="text-gray-400 text-xs font-medium uppercase tracking-wide">Actions</span>
        </div>

        <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3">
          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-2.5 text-red-400 text-xs">
              {error}
            </div>
          )}

          {/* Accept */}
          <button
            onClick={handleAccept}
            disabled={!activeItem || acting || !activeItem.predicted_path}
            className="flex items-center justify-center gap-2 h-10 w-full rounded-lg bg-green-600 hover:bg-green-700 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium transition-colors"
          >
            {acting && !correcting ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <CheckCircle2 size={14} />
            )}
            Accept
          </button>

          {/* Correct */}
          {!correcting ? (
            <button
              onClick={() => { setCorrecting(true); setCorrectionPath(activeItem?.predicted_path ?? '') }}
              disabled={!activeItem || acting}
              className="flex items-center justify-center gap-2 h-10 w-full rounded-lg bg-gray-800 hover:bg-gray-700 border border-gray-700 disabled:opacity-40 disabled:cursor-not-allowed text-gray-300 text-sm transition-colors"
            >
              Correct
            </button>
          ) : (
            <div className="flex flex-col gap-2">
              <select
                value={correctionPath}
                onChange={e => setCorrectionPath(e.target.value)}
                className="w-full rounded-lg bg-gray-800 border border-gray-700 text-gray-200 text-sm px-2.5 py-2 focus:outline-none focus:border-indigo-500"
              >
                <option value="">— choose folder —</option>
                {sortedPaths.map(p => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
              <div className="flex gap-2">
                <button
                  onClick={handleCorrect}
                  disabled={!correctionPath || acting}
                  className="flex-1 flex items-center justify-center gap-1.5 h-9 rounded-lg bg-indigo-500 hover:bg-indigo-600 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium transition-colors"
                >
                  {acting ? <Loader2 size={13} className="animate-spin" /> : null}
                  Confirm
                </button>
                <button
                  onClick={() => { setCorrecting(false); setCorrectionPath('') }}
                  disabled={acting}
                  className="flex items-center justify-center w-9 h-9 rounded-lg bg-gray-800 border border-gray-700 text-gray-400 hover:text-gray-200 transition-colors"
                >
                  <X size={14} />
                </button>
              </div>
            </div>
          )}

          {/* Skip */}
          <button
            onClick={handleSkip}
            disabled={!activeItem || acting}
            className="flex items-center justify-center h-9 w-full text-gray-500 hover:text-gray-300 text-sm disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Skip for now
          </button>
        </div>
      </div>

    </div>
  )
}
