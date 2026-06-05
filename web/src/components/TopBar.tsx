import { Check, X } from 'lucide-react'

export interface TopBarProps {
  authEmail: string | null
  wizardStep?: 'connect' | 'analyse' | 'review' | 'commit'
  section?: string
  draftSavedAt?: string | null
  onDisconnect?: () => void
}

const WIZARD_STEPS: Array<{ key: TopBarProps['wizardStep']; label: string }> = [
  { key: 'connect',  label: 'Connect'  },
  { key: 'analyse',  label: 'Analyse'  },
  { key: 'review',   label: 'Review'   },
  { key: 'commit',   label: 'Commit'   },
]

function formatSavedAt(iso: string): string {
  const diff = Math.floor((Date.now() - new Date(iso).getTime()) / 1000)
  if (diff < 10) return 'Saved just now'
  if (diff < 60) return `Saved ${diff}s ago`
  const mins = Math.floor(diff / 60)
  return `Saved ${mins} min ago`
}

function truncate(str: string, max: number): string {
  return str.length <= max ? str : str.slice(0, max) + '…'
}

export function TopBar({ authEmail, wizardStep, section, draftSavedAt, onDisconnect }: TopBarProps) {
  const activeIndex = WIZARD_STEPS.findIndex(s => s.key === wizardStep)

  const sectionLabel = wizardStep
    ? WIZARD_STEPS.find(s => s.key === wizardStep)?.label ?? ''
    : section ?? ''

  return (
    <header
      className="flex items-center justify-between px-4 shrink-0"
      style={{
        height: '52px',
        background: '#0f1117',
        borderBottom: '1px solid rgba(255,255,255,0.08)',
      }}
    >
      {/* LEFT — wordmark + section label */}
      <div className="flex items-center gap-2 min-w-0">
        <span className="text-white font-semibold text-sm tracking-tight whitespace-nowrap">
          DriveSort
        </span>
        {sectionLabel && (
          <>
            <span className="text-gray-600 text-sm select-none">/</span>
            <span className="text-gray-400 text-sm whitespace-nowrap">{sectionLabel}</span>
          </>
        )}
      </div>

      {/* CENTRE — wizard step indicator */}
      {wizardStep && (
        <nav aria-label="Setup progress" className="flex items-center gap-0 absolute left-1/2 -translate-x-1/2">
          {WIZARD_STEPS.map((step, i) => {
            const isActive    = i === activeIndex
            const isCompleted = i < activeIndex
            const isFuture    = i > activeIndex

            return (
              <div key={step.key} className="flex items-center">
                {/* connector line before (skip for first) */}
                {i > 0 && (
                  <div className="w-4 h-px bg-gray-700" />
                )}

                <div
                  className={[
                    'flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium whitespace-nowrap transition-colors',
                    isActive    ? 'bg-indigo-500 text-white'             : '',
                    isCompleted ? 'bg-gray-600 text-gray-300'            : '',
                    isFuture    ? 'bg-transparent text-gray-500'         : '',
                  ].join(' ')}
                >
                  {isCompleted && (
                    <Check className="w-3 h-3 shrink-0" strokeWidth={2.5} />
                  )}
                  {step.label}
                </div>
              </div>
            )
          })}
        </nav>
      )}

      {/* RIGHT — draft save indicator + auth chip */}
      <div className="flex items-center gap-3">
        {/* Draft save indicator */}
        {draftSavedAt && (
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-gray-400 animate-pulse" />
            <span className="text-gray-400 text-xs whitespace-nowrap">
              {formatSavedAt(draftSavedAt)}
            </span>
          </div>
        )}

        {/* Auth chip */}
        {authEmail ? (
          <div className="group flex items-center gap-1.5 bg-gray-800 rounded-full pl-2.5 pr-1.5 py-1">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 shrink-0" />
            <span className="text-gray-300 text-xs">
              {truncate(authEmail, 20)}
            </span>
            <button
              onClick={onDisconnect}
              aria-label="Disconnect Drive"
              className="opacity-0 group-hover:opacity-100 transition-opacity ml-0.5 text-gray-500 hover:text-gray-300 rounded-full p-0.5"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        ) : (
          <button
            className="text-xs font-medium text-indigo-400 border border-indigo-500 rounded-full px-3 py-1 hover:bg-indigo-500/10 transition-colors"
          >
            Connect Drive
          </button>
        )}
      </div>
    </header>
  )
}
