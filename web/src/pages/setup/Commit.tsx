import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  ChevronRight,
  File,
  Folder,
  FolderPlus,
  Loader2,
} from 'lucide-react'

import { api } from '@/lib/api'
import type { StageEntry, TaxonomyNode } from '@/lib/types'

// ---- Helpers ------------------------------------------------------------------

function nodeDepth(path: string): number {
  return path.split('/').length - 1
}

// ---- ColHeader ----------------------------------------------------------------

function ColHeader({ children }: { children: React.ReactNode }) {
  return (
    <div className="h-12 flex items-center px-4 border-b border-gray-800 text-gray-400 text-xs font-medium uppercase tracking-wide flex-shrink-0">
      {children}
    </div>
  )
}

// ---- NewBadge -----------------------------------------------------------------

function NewBadge() {
  return (
    <span className="bg-green-500/15 text-green-400 text-xs rounded px-1.5 py-0.5 flex-shrink-0">
      + new
    </span>
  )
}

// ---- TreeNode -----------------------------------------------------------------

interface TreeNodeProps {
  node: TaxonomyNode
  depth: number
  stage: StageEntry[]
  expandedPaths: Set<string>
  onToggle: (path: string) => void
}

function TreeNode({ node, depth, stage, expandedPaths, onToggle }: TreeNodeProps) {
  const children: TaxonomyNode[] = [] // children are rendered by parent iterating sorted list
  const filesForNode = stage.filter(s => s.target_folder_id === node.folder_id || s.target_path.startsWith(node.path + '/') || s.target_path === node.path)
  const isExpanded = expandedPaths.has(node.path)
  const hasFiles = filesForNode.length > 0
  const isNew = !node.folder_id

  return (
    <>
      {/* Node row */}
      <div
        className="flex items-center h-9 gap-2 px-3 hover:bg-gray-800/40 cursor-pointer select-none"
        style={{ paddingLeft: `${12 + depth * 16}px` }}
        onClick={() => onToggle(node.path)}
      >
        <ChevronRight
          size={14}
          className={[
            'flex-shrink-0 text-gray-500 transition-transform',
            hasFiles ? 'visible' : 'invisible',
            isExpanded ? 'rotate-90' : '',
          ].join(' ')}
        />
        <Folder size={14} className="flex-shrink-0 text-indigo-400" />
        <span className="text-gray-200 text-sm flex-1 min-w-0 truncate">{node.name}</span>
        {isNew && <NewBadge />}
      </div>

      {/* File rows */}
      {isExpanded &&
        filesForNode.map(entry => (
          <div
            key={entry.file_id}
            className="flex items-center h-8 gap-2 px-3 text-sm"
            style={{ paddingLeft: `${12 + (depth + 1) * 16}px` }}
          >
            <File size={12} className="flex-shrink-0 text-gray-600" />
            <span className="text-gray-400 text-sm flex-1 min-w-0 truncate">
              {entry.file_name || entry.file_id}
            </span>
            <span className="text-indigo-400 text-xs flex-shrink-0 truncate max-w-[140px]">
              &rarr; {entry.target_path}
            </span>
          </div>
        ))}
    </>
  )
}

// ---- ArchiveNode --------------------------------------------------------------

interface ArchiveNodeProps {
  files: StageEntry[]
  expanded: boolean
  onToggle: () => void
}

function ArchiveNode({ files, expanded, onToggle }: ArchiveNodeProps) {
  if (files.length === 0) return null
  return (
    <>
      <div
        className="flex items-center h-9 gap-2 px-3 hover:bg-gray-800/40 cursor-pointer select-none"
        style={{ paddingLeft: '12px' }}
        onClick={onToggle}
      >
        <ChevronRight
          size={14}
          className={[
            'flex-shrink-0 text-gray-500 transition-transform',
            expanded ? 'rotate-90' : '',
          ].join(' ')}
        />
        <Folder size={14} className="flex-shrink-0 text-indigo-400" />
        <span className="text-gray-200 text-sm flex-1">Archive</span>
        <NewBadge />
      </div>
      {expanded &&
        files.map(entry => (
          <div
            key={entry.file_id}
            className="flex items-center h-8 gap-2 px-3 text-sm"
            style={{ paddingLeft: `${12 + 16}px` }}
          >
            <File size={12} className="flex-shrink-0 text-gray-600" />
            <span className="text-gray-400 text-sm flex-1 min-w-0 truncate">
              {entry.file_name || entry.file_id}
            </span>
            <span className="text-indigo-400 text-xs flex-shrink-0 truncate max-w-[140px]">
              &rarr; {entry.target_path}
            </span>
          </div>
        ))}
    </>
  )
}

// ---- Commit (main) ------------------------------------------------------------

export function Commit() {
  const navigate = useNavigate()

  const [nodes, setNodes]                   = useState<TaxonomyNode[]>([])
  const [stage, setStage]                   = useState<StageEntry[]>([])
  const [expandedPaths, setExpandedPaths]   = useState<Set<string>>(new Set())
  const [archiveExpanded, setArchiveExpanded] = useState(false)
  const [loading, setLoading]               = useState(true)
  const [committing, setCommitting]         = useState(false)
  const [committed, setCommitted]           = useState(false)
  const [discardConfirm, setDiscardConfirm] = useState(false)
  const [error, setError]                   = useState<string | null>(null)

  // ---- Fetch on mount ---------------------------------------------------------
  useEffect(() => {
    Promise.all([api.taxonomy.list(), api.stage.list()])
      .then(([taxNodes, stageEntries]) => {
        setNodes(taxNodes)
        setStage(stageEntries)
      })
      .catch(() => setError('Failed to load data. Please try again.'))
      .finally(() => setLoading(false))
  }, [])

  // ---- Derived ----------------------------------------------------------------
  const newFolderCount = nodes.filter(n => !n.folder_id).length

  // Files whose target_folder_id doesn't match any taxonomy node folder_id
  const knownFolderIds = new Set(nodes.map(n => n.folder_id).filter(Boolean))
  const archiveFiles = stage.filter(s => !knownFolderIds.has(s.target_folder_id))

  // Unassigned = files not in stage at all, not in any member_ids
  const stagedFileIds = new Set(stage.map(s => s.file_id))
  const allMemberIds = new Set(nodes.flatMap(n => n.member_ids))
  const unassignedCount = [...allMemberIds].filter(id => !stagedFileIds.has(id)).length

  // ---- Tree toggle ------------------------------------------------------------
  function togglePath(path: string) {
    setExpandedPaths(prev => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }

  // ---- Actions ----------------------------------------------------------------
  async function handleCommit() {
    setCommitting(true)
    setError(null)
    try {
      await api.stage.commit()
      await api.draft.discard()
      setCommitted(true)
      navigate('/scan')
    } catch {
      setError('Commit failed. Please try again.')
      setCommitting(false)
    }
  }

  async function handleDiscard() {
    if (!discardConfirm) {
      setDiscardConfirm(true)
      return
    }
    try {
      await api.draft.discard()
      navigate('/setup/analyse')
    } catch {
      setError('Discard failed. Please try again.')
      setDiscardConfirm(false)
    }
  }

  // Reset discard confirm if user clicks away via another button
  function resetDiscardConfirm() {
    setDiscardConfirm(false)
  }

  // ---- Loading ----------------------------------------------------------------
  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 size={24} className="animate-spin text-gray-500" />
      </div>
    )
  }

  // ---- Empty taxonomy ---------------------------------------------------------
  if (nodes.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500 text-sm">
        No taxonomy built yet. Go back to the Review step.
      </div>
    )
  }

  // ---- Committed state (brief, before navigate fires) -------------------------
  if (committed) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-3">
        <CheckCircle2 size={32} className="text-green-500" />
        <span className="text-white text-lg font-medium">Changes committed to Drive</span>
      </div>
    )
  }

  // ---- Sorted nodes for tree rendering ----------------------------------------
  const sortedNodes = [...nodes].sort((a, b) => a.path.localeCompare(b.path))

  // Build a flat list rendering order: iterate sorted nodes, skipping those whose
  // ancestors are collapsed
  function isVisible(node: TaxonomyNode): boolean {
    if (!node.parent) return true
    if (!expandedPaths.has(node.parent)) return false
    const parentNode = nodes.find(n => n.path === node.parent)
    if (!parentNode) return false
    return isVisible(parentNode)
  }

  const visibleNodes = sortedNodes.filter(isVisible)

  // ---- Render -----------------------------------------------------------------
  return (
    <div className="h-full flex overflow-hidden">

      {/* ---- LEFT: Proposed Changes (60%) ---- */}
      <div className="w-[60%] flex flex-col border-r border-gray-800 overflow-hidden">
        <ColHeader>Proposed Changes</ColHeader>

        <div className="flex-1 overflow-y-auto">
          {visibleNodes.map(node => (
            <TreeNode
              key={node.path}
              node={node}
              depth={nodeDepth(node.path)}
              stage={stage}
              expandedPaths={expandedPaths}
              onToggle={togglePath}
            />
          ))}

          <ArchiveNode
            files={archiveFiles}
            expanded={archiveExpanded}
            onToggle={() => setArchiveExpanded(p => !p)}
          />
        </div>
      </div>

      {/* ---- RIGHT: Summary (40%) ---- */}
      <div className="w-[40%] flex flex-col overflow-hidden">
        <ColHeader>Summary</ColHeader>

        <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6">

          {/* Stats cards */}
          <div className="flex flex-col gap-3">
            {/* Folders to create */}
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex items-start gap-3">
              <FolderPlus size={18} className="text-indigo-400 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-white text-sm font-medium">{newFolderCount}</p>
                <p className="text-gray-500 text-xs">folders to create</p>
              </div>
            </div>

            {/* Files to move */}
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex items-start gap-3">
              <ArrowRight size={18} className="text-amber-400 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-white text-sm font-medium">{stage.length}</p>
                <p className="text-gray-500 text-xs">files to move</p>
              </div>
            </div>

            {/* Uncategorised */}
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex items-start gap-3">
              <AlertTriangle size={18} className="text-gray-400 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-white text-sm font-medium">{unassignedCount}</p>
                <p className="text-gray-500 text-xs">uncategorised</p>
              </div>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-3 text-red-400 text-sm">
              {error}
            </div>
          )}

          {/* Action buttons */}
          <div className="flex flex-col gap-3 mt-auto">
            {/* Commit to Drive */}
            <button
              onClick={() => { resetDiscardConfirm(); handleCommit() }}
              disabled={committing || stage.length === 0}
              className="flex items-center justify-center gap-2 h-11 w-full rounded-lg bg-indigo-500 hover:bg-indigo-600 disabled:opacity-40 disabled:cursor-not-allowed text-white font-medium transition-colors"
            >
              {committing ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Committing…
                </>
              ) : (
                'Commit to Drive \u2192'
              )}
            </button>

            {/* Back to edit */}
            <button
              onClick={() => { resetDiscardConfirm(); navigate('/setup/review') }}
              disabled={committing}
              className="flex items-center justify-center h-10 w-full rounded-lg bg-transparent border border-gray-700 hover:border-gray-500 disabled:opacity-40 disabled:cursor-not-allowed text-gray-300 text-sm transition-colors"
            >
              &larr; Back to edit
            </button>

            {/* Discard draft */}
            <button
              onClick={handleDiscard}
              disabled={committing}
              className="flex items-center justify-center h-9 w-full text-red-400 hover:text-red-300 text-sm disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {discardConfirm ? 'Tap again to confirm discard' : 'Discard draft'}
            </button>
          </div>
        </div>
      </div>

    </div>
  )
}
