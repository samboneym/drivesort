import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  DndContext,
  DragOverlay,
  useDraggable,
  useDroppable,
  type DragEndEvent,
} from '@dnd-kit/core'
import { CSS } from '@dnd-kit/utilities'
import {
  BarChart3,
  ChevronRight,
  Code,
  File,
  FileText,
  Folder,
  FolderPlus,
  Image,
  Pencil,
  Plus,
  Trash2,
} from 'lucide-react'

import { api } from '@/lib/api'
import type { AddNodePayload, DriveFileMeta, TaxonomyNode } from '@/lib/types'

// ---- Helpers ------------------------------------------------------------------

function fileIcon(mimeType: string) {
  if (
    mimeType.includes('document') ||
    mimeType.includes('pdf') ||
    mimeType.includes('text/plain') ||
    mimeType.includes('msword') ||
    mimeType.includes('wordprocessing')
  )
    return <FileText size={14} className="flex-shrink-0 text-gray-500" />

  if (mimeType.startsWith('image/'))
    return <Image size={14} className="flex-shrink-0 text-gray-500" />

  if (
    mimeType.includes('javascript') ||
    mimeType.includes('typescript') ||
    mimeType.includes('python') ||
    mimeType.includes('json') ||
    mimeType.includes('html') ||
    mimeType.includes('css') ||
    mimeType.includes('xml') ||
    mimeType.includes('x-sh')
  )
    return <Code size={14} className="flex-shrink-0 text-gray-500" />

  return <File size={14} className="flex-shrink-0 text-gray-500" />
}

function truncate(s: string, max: number) {
  return s.length > max ? s.slice(0, max) + '\u2026' : s
}

// ---- FileRow (draggable) -------------------------------------------------------

interface FileRowProps {
  file: DriveFileMeta
  isDragging: boolean
}

function FileRow({ file, isDragging }: FileRowProps) {
  const { attributes, listeners, setNodeRef, transform } = useDraggable({
    id: file.id,
  })

  const style = transform
    ? { transform: CSS.Translate.toString(transform) }
    : undefined

  return (
    <div
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
      className={[
        'flex items-center gap-3 px-4 py-2 cursor-grab select-none transition-opacity',
        isDragging ? 'opacity-50' : 'hover:bg-gray-800/50',
      ].join(' ')}
    >
      {fileIcon(file.mimeType)}
      <div className="min-w-0 flex-1">
        <p className="text-gray-200 text-sm leading-tight">
          {truncate(file.name, 36)}
        </p>
        <p className="text-gray-600 text-xs leading-tight mt-0.5">{file.mimeType}</p>
      </div>
    </div>
  )
}

// ---- TreeNode (droppable, recursive) ------------------------------------------

interface TreeNodeProps {
  node: TaxonomyNode
  depth: number
  allNodes: TaxonomyNode[]
  selectedPath: string | null
  expandedPaths: Set<string>
  renamingPath: string | null
  renameValue: string
  onSelect: (path: string) => void
  onToggleExpand: (path: string) => void
  onRenameStart: (path: string, currentName: string) => void
  onRenameChange: (val: string) => void
  onRenameCommit: (node: TaxonomyNode) => void
  onRenameCancel: () => void
  onAddChild: (parentPath: string) => void
  onDelete: (path: string) => void
  addingChildOf: string | null
  addingChildName: string
  onAddingChildNameChange: (val: string) => void
  onAddingChildCommit: (parentNode: TaxonomyNode) => void
  onAddingChildCancel: () => void
}

function TreeNode({
  node,
  depth,
  allNodes,
  selectedPath,
  expandedPaths,
  renamingPath,
  renameValue,
  onSelect,
  onToggleExpand,
  onRenameStart,
  onRenameChange,
  onRenameCommit,
  onRenameCancel,
  onAddChild,
  onDelete,
  addingChildOf,
  addingChildName,
  onAddingChildNameChange,
  onAddingChildCommit,
  onAddingChildCancel,
}: TreeNodeProps) {
  const { setNodeRef, isOver } = useDroppable({ id: node.path })
  const renameInputRef = useRef<HTMLInputElement>(null)
  const addInputRef = useRef<HTMLInputElement>(null)

  const children = allNodes.filter(n => n.parent === node.path)
  const hasChildren = children.length > 0
  const isExpanded = expandedPaths.has(node.path)
  const isSelected = selectedPath === node.path
  const isRenaming = renamingPath === node.path
  const isAddingChild = addingChildOf === node.path

  useEffect(() => {
    if (isRenaming && renameInputRef.current) {
      renameInputRef.current.focus()
      renameInputRef.current.select()
    }
  }, [isRenaming])

  useEffect(() => {
    if (isAddingChild && addInputRef.current) {
      addInputRef.current.focus()
    }
  }, [isAddingChild])

  return (
    <div>
      {/* Row */}
      <div
        ref={setNodeRef}
        className={[
          'group flex items-center h-9 px-3 gap-2 cursor-pointer select-none transition-colors',
          isOver ? 'bg-indigo-500/20' : isSelected ? 'bg-gray-800/70' : 'hover:bg-gray-800/50',
        ].join(' ')}
        style={{ paddingLeft: `${12 + depth * 16}px` }}
        onClick={() => onSelect(node.path)}
        onDoubleClick={e => {
          e.preventDefault()
          onRenameStart(node.path, node.name)
        }}
      >
        {/* Expand toggle */}
        <button
          className={[
            'flex-shrink-0 transition-transform text-gray-500 hover:text-gray-300',
            hasChildren ? 'visible' : 'invisible',
            isExpanded ? 'rotate-90' : '',
          ].join(' ')}
          style={{ width: 16, height: 16 }}
          onClick={e => {
            e.stopPropagation()
            onToggleExpand(node.path)
          }}
          tabIndex={-1}
          aria-label={isExpanded ? 'Collapse' : 'Expand'}
        >
          <ChevronRight size={14} />
        </button>

        {/* Folder icon */}
        <Folder size={14} className="flex-shrink-0 text-indigo-400" />

        {/* Name / rename input */}
        {isRenaming ? (
          <input
            ref={renameInputRef}
            value={renameValue}
            onChange={e => onRenameChange(e.target.value)}
            onClick={e => e.stopPropagation()}
            onKeyDown={e => {
              if (e.key === 'Enter') onRenameCommit(node)
              if (e.key === 'Escape') onRenameCancel()
            }}
            onBlur={() => onRenameCommit(node)}
            className="bg-transparent border-b border-indigo-500 text-white text-sm outline-none min-w-0 flex-1"
          />
        ) : (
          <span className="text-gray-200 text-sm flex-1 min-w-0 truncate">{node.name}</span>
        )}

        {/* Member count badge */}
        <span className="flex-shrink-0 bg-gray-800 text-gray-400 text-xs rounded px-1.5 py-0.5">
          {node.member_count}
        </span>

        {/* Action buttons (shown on row hover) */}
        <div className="hidden group-hover:flex items-center gap-1 flex-shrink-0">
          <button
            title="Add subfolder"
            onClick={e => {
              e.stopPropagation()
              onAddChild(node.path)
            }}
            className="text-gray-400 hover:text-white p-0.5 rounded"
          >
            <Plus size={13} />
          </button>
          <button
            title="Rename"
            onClick={e => {
              e.stopPropagation()
              onRenameStart(node.path, node.name)
            }}
            className="text-gray-400 hover:text-white p-0.5 rounded"
          >
            <Pencil size={13} />
          </button>
          <button
            title="Delete"
            onClick={e => {
              e.stopPropagation()
              onDelete(node.path)
            }}
            className="text-red-400 hover:text-red-300 p-0.5 rounded"
          >
            <Trash2 size={13} />
          </button>
        </div>
      </div>

      {/* Inline add-child input */}
      {isAddingChild && (
        <div
          className="flex items-center h-9 px-3 gap-2"
          style={{ paddingLeft: `${12 + (depth + 1) * 16 + 16 + 8}px` }}
        >
          <Folder size={14} className="flex-shrink-0 text-indigo-400/50" />
          <input
            ref={addInputRef}
            value={addingChildName}
            placeholder="Folder name"
            onChange={e => onAddingChildNameChange(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter') onAddingChildCommit(node)
              if (e.key === 'Escape') onAddingChildCancel()
            }}
            onBlur={() => {
              if (addingChildName.trim()) onAddingChildCommit(node)
              else onAddingChildCancel()
            }}
            className="bg-transparent border-b border-indigo-500 text-white text-sm outline-none min-w-0 flex-1 placeholder-gray-600"
          />
        </div>
      )}

      {/* Children */}
      {isExpanded &&
        children.map(child => (
          <TreeNode
            key={child.path}
            node={child}
            depth={depth + 1}
            allNodes={allNodes}
            selectedPath={selectedPath}
            expandedPaths={expandedPaths}
            renamingPath={renamingPath}
            renameValue={renameValue}
            onSelect={onSelect}
            onToggleExpand={onToggleExpand}
            onRenameStart={onRenameStart}
            onRenameChange={onRenameChange}
            onRenameCommit={onRenameCommit}
            onRenameCancel={onRenameCancel}
            onAddChild={onAddChild}
            onDelete={onDelete}
            addingChildOf={addingChildOf}
            addingChildName={addingChildName}
            onAddingChildNameChange={onAddingChildNameChange}
            onAddingChildCommit={onAddingChildCommit}
            onAddingChildCancel={onAddingChildCancel}
          />
        ))}
    </div>
  )
}

// ---- Review (main) ------------------------------------------------------------

interface ReviewProps {
  onDraftSave: (isoTimestamp: string) => void
}

export function Review({ onDraftSave }: ReviewProps) {
  const navigate = useNavigate()

  const [nodes, setNodes]               = useState<TaxonomyNode[]>([])
  const [files, setFiles]               = useState<DriveFileMeta[]>([])
  const [selectedPath, setSelectedPath] = useState<string | null>(null)
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(new Set())
  const [renamingPath, setRenamingPath] = useState<string | null>(null)
  const [renameValue, setRenameValue]   = useState('')
  const [loading, setLoading]           = useState(true)

  // Add-child inline state
  const [addingChildOf, setAddingChildOf]     = useState<string | null>(null)
  const [addingChildName, setAddingChildName] = useState('')
  // Add-root inline state
  const [addingRoot, setAddingRoot]   = useState(false)
  const [addingRootName, setAddingRootName] = useState('')
  const addRootInputRef = useRef<HTMLInputElement>(null)

  // Drag active file id for DragOverlay
  const [activeFileId, setActiveFileId] = useState<string | null>(null)

  // Auto-save debounce
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // ---- Fetch on mount --------------------------------------------------------
  useEffect(() => {
    Promise.all([api.taxonomy.list(), api.files.list()])
      .then(([taxonomyNodes, driveFiles]) => {
        setNodes(taxonomyNodes)
        setFiles(driveFiles)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (addingRoot && addRootInputRef.current) {
      addRootInputRef.current.focus()
    }
  }, [addingRoot])

  // ---- Auto-save -------------------------------------------------------------
  const triggerAutoSave = useCallback(
    (latestNodes: TaxonomyNode[]) => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
      saveTimerRef.current = setTimeout(() => {
        const payload = {
          taxonomy_tree: Object.fromEntries(latestNodes.map(n => [n.path, n])),
          staged_changes: [] as [],
          user_decisions: [] as [],
        }
        api.draft.save(payload)
          .then(() => onDraftSave(new Date().toISOString()))
          .catch(() => {})
      }, 1500)
    },
    [onDraftSave],
  )

  // ---- Derived ---------------------------------------------------------------
  const selectedNode = nodes.find(n => n.path === selectedPath) ?? null

  const assignedFileIds = new Set(nodes.flatMap(n => n.member_ids))
  const unassignedCount = files.filter(f => !assignedFileIds.has(f.id)).length

  const selectedFiles = selectedNode
    ? files.filter(f => selectedNode.member_ids.includes(f.id))
    : []

  // ---- Tree helpers ----------------------------------------------------------
  function toggleExpand(path: string) {
    setExpandedPaths(prev => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }

  // ---- Rename ----------------------------------------------------------------
  function handleRenameCommit(node: TaxonomyNode) {
    const trimmed = renameValue.trim()
    setRenamingPath(null)
    if (!trimmed || trimmed === node.name) return

    const updated = nodes.map(n =>
      n.path === node.path ? { ...n, name: trimmed } : n,
    )
    setNodes(updated)
    triggerAutoSave(updated)

    api.taxonomy.patch(node.path, { name: trimmed }).catch(() => {})
  }

  // ---- Delete ----------------------------------------------------------------
  function handleDelete(path: string) {
    // Remove node and all descendants
    const toRemove = new Set<string>()
    function collect(p: string) {
      toRemove.add(p)
      nodes.filter(n => n.parent === p).forEach(n => collect(n.path))
    }
    collect(path)

    const updated = nodes.filter(n => !toRemove.has(n.path))
    setNodes(updated)
    if (selectedPath && toRemove.has(selectedPath)) setSelectedPath(null)
    triggerAutoSave(updated)

    api.taxonomy.delete(path).catch(() => {})
  }

  // ---- Add child -------------------------------------------------------------
  function handleAddChildCommit(parentNode: TaxonomyNode) {
    const name = addingChildName.trim()
    setAddingChildOf(null)
    setAddingChildName('')
    if (!name) return

    const newPath = `${parentNode.path}/${name}`
    const payload: AddNodePayload = {
      path: newPath,
      name,
      parent: parentNode.path,
      member_embeddings: [],
      member_ids: [],
    }

    const optimistic: TaxonomyNode = {
      path: newPath,
      name,
      parent: parentNode.path,
      centroid: [],
      member_ids: [],
      member_count: 0,
      folder_id: '',
      description: '',
    }

    const updated = [...nodes, optimistic]
    setNodes(updated)
    setExpandedPaths(prev => new Set([...prev, parentNode.path]))
    triggerAutoSave(updated)

    api.taxonomy.add(payload)
      .then(created => {
        setNodes(prev =>
          prev.map(n => (n.path === newPath ? created : n)),
        )
      })
      .catch(() => {})
  }

  // ---- Add root --------------------------------------------------------------
  function handleAddRootCommit() {
    const name = addingRootName.trim()
    setAddingRoot(false)
    setAddingRootName('')
    if (!name) return

    const newPath = name
    const payload: AddNodePayload = {
      path: newPath,
      name,
      parent: null,
      member_embeddings: [],
      member_ids: [],
    }

    const optimistic: TaxonomyNode = {
      path: newPath,
      name,
      parent: null,
      centroid: [],
      member_ids: [],
      member_count: 0,
      folder_id: '',
      description: '',
    }

    const updated = [...nodes, optimistic]
    setNodes(updated)
    triggerAutoSave(updated)

    api.taxonomy.add(payload)
      .then(created => {
        setNodes(prev =>
          prev.map(n => (n.path === newPath ? created : n)),
        )
      })
      .catch(() => {})
  }

  // ---- DnD -------------------------------------------------------------------
  function handleDragEnd(event: DragEndEvent) {
    const fileId = event.active.id as string
    const targetPath = event.over?.id as string | undefined
    setActiveFileId(null)

    if (!targetPath) return

    const targetNode = nodes.find(n => n.path === targetPath)
    if (!targetNode) return

    // Find which node currently owns this file
    const updated = nodes.map(n => {
      if (n.member_ids.includes(fileId) && n.path !== targetPath) {
        return { ...n, member_ids: n.member_ids.filter(id => id !== fileId), member_count: n.member_count - 1 }
      }
      if (n.path === targetPath && !n.member_ids.includes(fileId)) {
        return { ...n, member_ids: [...n.member_ids, fileId], member_count: n.member_count + 1 }
      }
      return n
    })

    setNodes(updated)
    triggerAutoSave(updated)

    api.taxonomy.confirm(targetPath, { file_id: fileId, embedding: [] }).catch(() => {})
  }

  const activeFile = activeFileId ? files.find(f => f.id === activeFileId) : null

  // ---- Column header ---------------------------------------------------------
  function ColHeader({ children }: { children: React.ReactNode }) {
    return (
      <div className="h-12 flex items-center px-4 border-b border-gray-800 text-gray-400 text-xs font-medium uppercase tracking-wide flex-shrink-0">
        {children}
      </div>
    )
  }

  // ---- Loading ---------------------------------------------------------------
  if (loading) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500 text-sm">
        Loading…
      </div>
    )
  }

  // ---- Empty taxonomy --------------------------------------------------------
  if (nodes.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500 text-sm">
        No taxonomy yet — go back to re-run analysis.
      </div>
    )
  }

  const rootNodes = nodes.filter(n => n.parent === null)

  // ---- Render ----------------------------------------------------------------
  return (
    <DndContext
      onDragStart={event => setActiveFileId(event.active.id as string)}
      onDragEnd={handleDragEnd}
      onDragCancel={() => setActiveFileId(null)}
    >
      <div className="h-full flex overflow-hidden">
        {/* ---- LEFT: Visualization ---- */}
        <div className="w-[30%] flex flex-col border-r border-gray-800 overflow-hidden">
          <ColHeader>Visualization</ColHeader>

          <div className="flex-1 flex flex-col overflow-y-auto p-4 gap-4">
            {/* Placeholder card */}
            <div className="flex-1 bg-gray-900 border border-gray-800 rounded-xl flex flex-col items-center justify-center gap-3 min-h-[200px]">
              <BarChart3 size={32} className="text-gray-600" />
              <span className="text-gray-600 text-sm">UMAP scatter coming soon</span>
            </div>

            {/* Summary chips */}
            <div className="flex flex-wrap gap-2">
              <span className="bg-gray-900 border border-gray-800 rounded-md px-2 py-1 text-gray-400 text-xs">
                {nodes.length} folders
              </span>
              <span className="bg-gray-900 border border-gray-800 rounded-md px-2 py-1 text-gray-400 text-xs">
                {files.length} files
              </span>
              <span className="bg-gray-900 border border-gray-800 rounded-md px-2 py-1 text-gray-400 text-xs">
                {unassignedCount} unassigned
              </span>
            </div>
          </div>
        </div>

        {/* ---- MIDDLE: Files ---- */}
        <div className="w-[35%] flex flex-col border-r border-gray-800 overflow-hidden">
          <ColHeader>
            <span>Files</span>
            {selectedNode && (
              <span className="ml-1 normal-case font-normal text-gray-500">
                &mdash; {selectedNode.name} ({selectedNode.member_count})
              </span>
            )}
          </ColHeader>

          <div className="flex-1 overflow-y-auto">
            {!selectedNode ? (
              <div className="h-full flex items-center justify-center text-gray-600 text-sm">
                Select a folder to see its files
              </div>
            ) : (
              <div className="flex flex-col">
                {selectedFiles.map(file => (
                  <FileRow
                    key={file.id}
                    file={file}
                    isDragging={activeFileId === file.id}
                  />
                ))}
                {selectedFiles.length === 0 && (
                  <div className="px-4 py-6 text-gray-600 text-sm text-center">
                    No files in this folder
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Sticky commit button */}
          <div className="flex-shrink-0 p-3 border-t border-gray-800">
            <button
              onClick={() => navigate('/setup/commit')}
              className="w-full bg-indigo-500 hover:bg-indigo-600 text-white text-sm font-medium h-9 rounded-lg transition-colors"
            >
              Proceed to Commit &rarr;
            </button>
          </div>
        </div>

        {/* ---- RIGHT: Taxonomy ---- */}
        <div className="w-[35%] flex flex-col overflow-hidden">
          <div className="h-12 flex items-center px-4 border-b border-gray-800 flex-shrink-0">
            <span className="text-gray-400 text-xs font-medium uppercase tracking-wide flex-1">
              Taxonomy
            </span>
            <button
              title="Add root folder"
              onClick={() => setAddingRoot(true)}
              className="text-gray-400 hover:text-white transition-colors"
            >
              <FolderPlus size={15} />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto">
            {rootNodes.map(node => (
              <TreeNode
                key={node.path}
                node={node}
                depth={0}
                allNodes={nodes}
                selectedPath={selectedPath}
                expandedPaths={expandedPaths}
                renamingPath={renamingPath}
                renameValue={renameValue}
                onSelect={setSelectedPath}
                onToggleExpand={toggleExpand}
                onRenameStart={(path, name) => {
                  setRenamingPath(path)
                  setRenameValue(name)
                }}
                onRenameChange={setRenameValue}
                onRenameCommit={handleRenameCommit}
                onRenameCancel={() => setRenamingPath(null)}
                onAddChild={path => {
                  setAddingChildOf(path)
                  setAddingChildName('')
                }}
                onDelete={handleDelete}
                addingChildOf={addingChildOf}
                addingChildName={addingChildName}
                onAddingChildNameChange={setAddingChildName}
                onAddingChildCommit={handleAddChildCommit}
                onAddingChildCancel={() => {
                  setAddingChildOf(null)
                  setAddingChildName('')
                }}
              />
            ))}

            {/* Add root inline input */}
            {addingRoot && (
              <div className="flex items-center h-9 px-3 gap-2">
                <Folder size={14} className="flex-shrink-0 text-indigo-400/50" />
                <input
                  ref={addRootInputRef}
                  value={addingRootName}
                  placeholder="Folder name"
                  onChange={e => setAddingRootName(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter') handleAddRootCommit()
                    if (e.key === 'Escape') {
                      setAddingRoot(false)
                      setAddingRootName('')
                    }
                  }}
                  onBlur={() => {
                    if (addingRootName.trim()) handleAddRootCommit()
                    else {
                      setAddingRoot(false)
                      setAddingRootName('')
                    }
                  }}
                  className="bg-transparent border-b border-indigo-500 text-white text-sm outline-none min-w-0 flex-1 placeholder-gray-600"
                />
              </div>
            )}

            {/* Add root folder button */}
            <div className="p-3">
              <button
                onClick={() => setAddingRoot(true)}
                className="w-full h-8 border border-dashed border-gray-700 rounded-lg text-gray-500 hover:text-gray-300 hover:border-gray-500 text-sm transition-colors"
              >
                + Add root folder
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* DragOverlay ghost */}
      <DragOverlay>
        {activeFile ? (
          <div className="flex items-center gap-3 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg shadow-xl opacity-90">
            {fileIcon(activeFile.mimeType)}
            <span className="text-gray-200 text-sm">{truncate(activeFile.name, 36)}</span>
          </div>
        ) : null}
      </DragOverlay>
    </DndContext>
  )
}
