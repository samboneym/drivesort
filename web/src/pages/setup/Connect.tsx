import { useState } from 'react'
import { HardDrive, CheckCircle2, Chrome, Loader2 } from 'lucide-react'
import { api } from '@/lib/api'

interface ConnectProps {
  onAuth: (email: string) => void
}

export function Connect({ onAuth: _onAuth }: ConnectProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleConnect() {
    setLoading(true)
    setError(null)
    try {
      const { auth_url } = await api.auth.login()
      window.location.href = auth_url
    } catch {
      setError('Failed to start the OAuth flow. Please try again.')
      setLoading(false)
    }
  }

  return (
    <div className="flex items-center justify-center h-full w-full bg-[#0f1117]">
      <div className="bg-gray-900 border border-gray-800 rounded-2xl max-w-md w-full p-10">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="rounded-lg bg-indigo-500/15 p-2">
            <HardDrive size={28} className="text-indigo-500" />
          </div>
          <span className="text-white font-semibold text-xl">DriveSort</span>
        </div>

        {/* Heading */}
        <h1 className="text-white text-2xl font-semibold mt-8">
          Connect your Google Drive
        </h1>

        {/* Subtext */}
        <p className="text-gray-400 text-sm mt-2 leading-relaxed">
          DriveSort analyses your files locally and builds a folder taxonomy — nothing leaves your machine.
        </p>

        {/* Feature list */}
        <ul className="mt-6 flex flex-col gap-3">
          {[
            'Reads file names and metadata only — no file contents uploaded',
            'All AI processing runs on your local machine (Ollama)',
            'Drive writes only happen when you explicitly commit changes',
          ].map((item) => (
            <li key={item} className="flex items-start gap-2">
              <CheckCircle2 size={14} className="text-indigo-400 mt-0.5 shrink-0" />
              <span className="text-gray-300 text-sm">{item}</span>
            </li>
          ))}
        </ul>

        {/* Connect button */}
        <div className="mt-8">
          <button
            onClick={handleConnect}
            disabled={loading}
            className="flex items-center justify-center gap-2 w-full bg-indigo-500 hover:bg-indigo-600 disabled:opacity-60 disabled:cursor-not-allowed text-white font-medium rounded-lg h-11 transition-colors"
          >
            {loading ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <Chrome size={16} />
            )}
            Connect Google Drive
          </button>

          {error && (
            <p className="text-red-400 text-xs mt-2">{error}</p>
          )}
        </div>

        {/* Footer note */}
        <p className="text-gray-600 text-xs mt-6 text-center">
          OAuth flow opens in this tab · Redirects back to localhost:7432
        </p>
      </div>
    </div>
  )
}
