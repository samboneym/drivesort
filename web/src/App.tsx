import { useEffect, useState } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { api } from '@/lib/api'

// Pages — replaced one-by-one as v0 output arrives
import { Connect } from '@/pages/setup/Connect'
import { Analyse } from '@/pages/setup/Analyse'
import { Review }  from '@/pages/setup/Review'
import { Commit }  from '@/pages/setup/Commit'
import { Scan }    from '@/pages/Scan'
import { Status }  from '@/pages/Status'

export default function App() {
  const [authEmail, setAuthEmail]   = useState<string | null>(null)
  const [_draftSaved, setDraftSaved] = useState<string | null>(null)

  useEffect(() => {
    api.auth.status()
      .then(s => setAuthEmail(s.authenticated ? s.email : null))
      .catch(() => {})
  }, [])

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-[#0f1117]">
      {/* TopBar slot — filled after v0 prompt #1 */}
      <div id="topbar-slot" className="shrink-0" />
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
