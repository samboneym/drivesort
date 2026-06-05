import { useEffect, useState } from 'react'
import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { api } from '@/lib/api'
import { TopBar } from '@/components/TopBar'
import type { TopBarProps } from '@/components/TopBar'

// Pages — replaced one-by-one as v0 output arrives
import { Connect } from '@/pages/setup/Connect'
import { Analyse } from '@/pages/setup/Analyse'
import { Review }  from '@/pages/setup/Review'
import { Commit }  from '@/pages/setup/Commit'
import { Scan }    from '@/pages/Scan'
import { Status }  from '@/pages/Status'

function deriveTopBarRoute(pathname: string): Pick<TopBarProps, 'wizardStep' | 'section'> {
  if (pathname.startsWith('/setup/connect'))  return { wizardStep: 'connect'  }
  if (pathname.startsWith('/setup/analyse'))  return { wizardStep: 'analyse'  }
  if (pathname.startsWith('/setup/review'))   return { wizardStep: 'review'   }
  if (pathname.startsWith('/setup/commit'))   return { wizardStep: 'commit'   }
  if (pathname.startsWith('/scan'))           return { section: 'Scan'        }
  if (pathname.startsWith('/status'))         return { section: 'Status'      }
  return {}
}

function AppShell() {
  const [authEmail, setAuthEmail]   = useState<string | null>(null)
  const [draftSaved, setDraftSaved] = useState<string | null>(null)
  const location = useLocation()

  useEffect(() => {
    api.auth.status()
      .then(s => setAuthEmail(s.authenticated ? s.email : null))
      .catch(() => {})
  }, [])

  const { wizardStep, section } = deriveTopBarRoute(location.pathname)

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-[#0f1117]">
      <TopBar
        authEmail={authEmail}
        wizardStep={wizardStep}
        section={section}
        draftSavedAt={draftSaved}
        onDisconnect={() => setAuthEmail(null)}
      />
      <div className="flex-1 overflow-hidden">
        <Routes>
          <Route path="/" element={<Navigate to={authEmail ? '/setup/analyse' : '/setup/connect'} replace />} />
          <Route path="/setup/connect" element={<Connect onAuth={setAuthEmail} />} />
          <Route path="/setup/analyse" element={<Analyse />} />
          <Route path="/setup/review"  element={<Review onDraftSave={setDraftSaved} />} />
          <Route path="/setup/commit"  element={<Commit />} />
          <Route path="/scan"          element={<Scan />} />
          <Route path="/status"        element={<Status />} />
          <Route path="*"              element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </div>
  )
}

export default function App() {
  return <AppShell />
}
