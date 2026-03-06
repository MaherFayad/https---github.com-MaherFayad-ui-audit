import { useState, useRef } from 'react'
import {
  Link, Upload, ImagePlus, Monitor, Laptop, Tablet, Smartphone,
  Search, Flame, Eye, MousePointer2, Mouse, Accessibility, Sparkles,
  ArrowRight, ArrowLeft, RefreshCw, LayoutTemplate
} from 'lucide-react'
import './App.css'

function App() {
  const [step, setStep] = useState(1) // 1: Source, 2: Layout, 3: Analysis
  const [mode, setMode] = useState('url') // 'url' or 'upload'
  const [url, setUrl] = useState('')
  const [viewportWidth, setViewportWidth] = useState(1920)
  const [context, setContext] = useState('')
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState('')
  const [results, setResults] = useState(null)
  const [activeTab, setActiveTab] = useState('attention')
  const [error, setError] = useState('')
  const fileRef = useRef(null)
  const [fileName, setFileName] = useState('')

  // The specific widths requested by the user, now with associated icons
  const viewports = [
    { width: 1920, label: 'Desktop', icon: <Monitor size={24} /> },
    { width: 1440, label: 'Laptop', icon: <Laptop size={24} /> },
    { width: 768, label: 'Tablet', icon: <Tablet size={24} /> },
    { width: 390, label: 'Mobile', icon: <Smartphone size={24} /> }
  ]

  const handleAnalyze = async () => {
    setLoading(true)
    setError('')
    setResults(null)
    setStep(3)
    setProgress('Starting analysis...')

    try {
      const formData = new FormData()

      if (mode === 'url') {
        if (!url.trim()) { setError('Please enter a URL'); setStep(1); setLoading(false); return }
        formData.append('url', url)
      } else {
        const file = fileRef.current?.files?.[0]
        if (!file) { setError('Please select an image'); setStep(1); setLoading(false); return }
        formData.append('image', file)
      }

      let deviceType = 'desktop'
      if (viewportWidth <= 390) deviceType = 'mobile'
      else if (viewportWidth <= 1024) deviceType = 'tablet'

      formData.append('device_type', deviceType)
      formData.append('viewport_width', viewportWidth.toString())
      formData.append('context', context)

      setProgress('Capturing & analyzing... This may take 30-60 seconds.')

      const res = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        body: formData
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.error || 'Analysis failed')
      }

      const data = await res.json()
      setResults(data)
      setActiveTab('attention')
      setProgress('')
    } catch (e) {
      setError(e.message)
      setProgress('')
    } finally {
      setLoading(false)
    }
  }

  const tabs = [
    { id: 'attention', label: 'Attention Heatmap', icon: <Flame size={16} /> },
    { id: 'scanpath', label: 'Scanpath', icon: <Eye size={16} /> },
    { id: 'scroll', label: 'Scroll Depth', icon: <MousePointer2 size={16} /> },
    ...(results?.mouse_movement ? [{ id: 'mouse', label: 'Mouse Movement', icon: <Mouse size={16} /> }] : []),
    { id: 'accessibility', label: 'Accessibility', icon: <Accessibility size={16} /> },
    ...(results?.ux_overview ? [{ id: 'ux_overview', label: 'UX Overview', icon: <Sparkles size={16} /> }] : []),
  ]

  const tabImages = {
    attention: results?.attention,
    scanpath: results?.scanpath,
    scroll: results?.scroll_depth,
    mouse: results?.mouse_movement,
  }

  // Calculate stats for the results dashboard (Colored only for success/warning)
  const stats = results ? [
    { label: 'Focus Score', value: `${results.focus_score?.toFixed(1)}%`, color: results.focus_score >= 60 ? 'text-green-600' : 'text-amber-500' },
    { label: 'Clarity Score', value: `${results.clarity_score?.toFixed(1)}%`, color: results.clarity_score >= 60 ? 'text-green-600' : 'text-amber-500' },
    { label: 'Elements Found', value: results.boxes?.length, color: 'text-gray-800' },
    { label: 'Above Fold', value: `${results.above_fold_analysis?.above_fold_attention_pct?.toFixed(0)}%`, color: 'text-gray-800' }
  ] : []

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <LayoutTemplate className="logo-icon" size={24} strokeWidth={2.5} />
            <h1>UI Analyzer</h1>
          </div>
        </div>
      </header>

      <main className="main">
        {/* The Notion-style Two-Column Layout for Input Phase */}
        {!results && (
          <div className="wizard-layout">

            {/* Vertical Sidebar Stepper */}
            <aside className="wizard-sidebar">
              <div className="sidebar-sticky">
                <div className="sidebar-title">Analysis Setup</div>
                <nav className="stepper-nav">
                  <div className={`stepper-item ${step === 1 ? 'active' : step > 1 ? 'completed' : ''}`} onClick={() => step > 1 && setStep(1)}>
                    <div className="stepper-badge">1</div>
                    <span className="stepper-label">Input Source</span>
                  </div>
                  <div className="stepper-connector"></div>

                  <div className={`stepper-item ${step === 2 ? 'active' : step > 2 ? 'completed' : ''}`}>
                    <div className="stepper-badge">2</div>
                    <span className="stepper-label">Viewport Layout</span>
                  </div>
                  <div className="stepper-connector"></div>

                  <div className={`stepper-item ${step === 3 ? 'active' : ''}`}>
                    <div className="stepper-badge">3</div>
                    <span className="stepper-label">AI Analysis</span>
                  </div>
                </nav>
              </div>
            </aside>

            {/* Main Content Area */}
            <div className="wizard-content">
              {error && <div className="error-banner"><Flame size={18} /> {error}</div>}

              {/* Step 1: Input Source */}
              {step === 1 && (
                <div className="pane fade-in">
                  <h2 className="pane-title">Choose your source</h2>
                  <p className="pane-desc">Enter a public URL or upload a local design mockup.</p>

                  <div className="mode-toggle">
                    <button className={`toggle-btn ${mode === 'url' ? 'active' : ''}`} onClick={() => setMode('url')}>
                      <Link size={16} /> Website URL
                    </button>
                    <button className={`toggle-btn ${mode === 'upload' ? 'active' : ''}`} onClick={() => setMode('upload')}>
                      <Upload size={16} /> Upload Image
                    </button>
                  </div>

                  <div className="card">
                    {mode === 'url' ? (
                      <div className="field">
                        <label>Public URL</label>
                        <input
                          type="url"
                          placeholder="https://example.com"
                          value={url}
                          onChange={e => setUrl(e.target.value)}
                          className="input"
                          autoFocus
                        />
                      </div>
                    ) : (
                      <div className="field">
                        <label>Mockup Image</label>
                        <div className="upload-zone" onClick={() => fileRef.current?.click()}>
                          <input
                            ref={fileRef}
                            type="file"
                            accept="image/*"
                            hidden
                            onChange={e => setFileName(e.target.files?.[0]?.name || '')}
                          />
                          <ImagePlus size={32} className="upload-icon" />
                          <span className="upload-text">{fileName || 'Click or drag image to upload'}</span>
                        </div>
                      </div>
                    )}

                    <div className="field mt-6">
                      <label className="flex-between">
                        <span>Page Context <span className="badge-optional">Optional</span></span>
                      </label>
                      <textarea
                        placeholder="E.g., A SaaS landing page targeting small businesses. Needs to drive free trial signups."
                        value={context}
                        onChange={e => setContext(e.target.value)}
                        rows={3}
                        className="textarea"
                      />
                      <p className="field-help">Adding context enables the Gemini UX Overview report.</p>
                    </div>
                  </div>

                  <div className="pane-footer">
                    <button
                      className="btn btn-primary"
                      onClick={() => {
                        if (mode === 'url' && !url) setError('Please enter a website URL.')
                        else if (mode === 'upload' && !fileName) setError('Please select an image file.')
                        else { setError(''); setStep(2) }
                      }}
                    >
                      Continue <ArrowRight size={16} />
                    </button>
                  </div>
                </div>
              )}

              {/* Step 2: Viewport Selection */}
              {step === 2 && (
                <div className="pane slide-in-right">
                  <h2 className="pane-title">Select layout</h2>
                  <p className="pane-desc">Choose the device width to simulate for the fold line calculation.</p>

                  <div className="viewport-cards">
                    {viewports.map(vp => (
                      <div
                        key={vp.width}
                        className={`vp-card ${viewportWidth === vp.width ? 'active' : ''}`}
                        onClick={() => setViewportWidth(vp.width)}
                      >
                        <div className="vp-icon">{vp.icon}</div>
                        <div className="vp-info">
                          <div className="vp-width">{vp.width}</div>
                          <div className="vp-label">{vp.label}</div>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="pane-footer mt-8">
                    <button className="btn btn-ghost" onClick={() => setStep(1)}>
                      <ArrowLeft size={16} /> Back
                    </button>
                    <button className="btn btn-primary" onClick={handleAnalyze} disabled={loading}>
                      <Search size={16} /> Run Analysis
                    </button>
                  </div>
                </div>
              )}

              {/* Step 3: Loading */}
              {step === 3 && loading && (
                <div className="pane fade-in flex-center">
                  <div className="loading-state">
                    <RefreshCw size={32} className="spinner" />
                    <h3>{progress}</h3>
                    <p className="text-muted">Running EML-NET and OmniParser V2...</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Results View (Full Width) */}
        {results && (
          <div className="results-panel fade-in">
            <div className="results-header">
              <button className="btn btn-ghost" onClick={() => { setResults(null); setStep(1); }}>
                <ArrowLeft size={16} /> New Analysis
              </button>
              <h2 className="results-title">Analysis Complete</h2>
            </div>

            <div className="stats-grid">
              {stats.map(s => (
                <div key={s.label} className="stat-card">
                  <div className="stat-label">{s.label}</div>
                  <div className={`stat-value ${s.color}`}>{s.value}</div>
                </div>
              ))}
            </div>

            <div className="tabs-container">
              <div className="tabs-list">
                {tabs.map(tab => (
                  <button
                    key={tab.id}
                    className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
                    onClick={() => setActiveTab(tab.id)}
                  >
                    {tab.icon} {tab.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="tab-content">
              {tabImages[activeTab] && (
                <div className="image-viewer">
                  <img src={tabImages[activeTab]} alt={activeTab} />
                </div>
              )}

              {activeTab === 'accessibility' && (
                <div className="report-markdown" dangerouslySetInnerHTML={{ __html: markdownToHtml(results.accessibility_report || '') }} />
              )}

              {activeTab === 'ux_overview' && results.ux_overview && (
                <div className="report-markdown" dangerouslySetInnerHTML={{ __html: markdownToHtml(results.ux_overview) }} />
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

function markdownToHtml(md) {
  if (!md) return ''
  return md
    .replace(/### (.*)/g, '<h3>$1</h3>')
    .replace(/## (.*)/g, '<h2>$1</h2>')
    .replace(/# (.*)/g, '<h1>$1</h1>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code>$1</code>')
    .replace(/^- (.*)/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
    .replace(/\n\n/g, '<br/><br/>')
    .replace(/\n/g, '<br/>')
}

export default App
