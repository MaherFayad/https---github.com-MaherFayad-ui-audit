import { useState, useRef, useEffect, useCallback } from 'react'
import {
  Link, Upload, ImagePlus, Monitor, Laptop, Tablet, Smartphone,
  Search, Flame, Eye, MousePointer2, Mouse, Accessibility, Sparkles,
  ArrowRight, ArrowLeft, RefreshCw, LayoutTemplate, Download, FileText,
  Image as ImageIcon, ChevronDown
} from 'lucide-react'
import { marked } from 'marked'
import html2canvas from 'html2canvas'
import { jsPDF } from 'jspdf'
import './App.css'

function App() {
  const [step, setStep] = useState(1)
  const [mode, setMode] = useState('url')
  const [url, setUrl] = useState('')
  const [viewportWidth, setViewportWidth] = useState(1920)
  const [context, setContext] = useState('')
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState('')
  const [results, setResults] = useState(null)
  const [activeTab, setActiveTab] = useState('original')
  const [error, setError] = useState('')
  const fileRef = useRef(null)
  const [fileName, setFileName] = useState('')
  const [showExportMenu, setShowExportMenu] = useState(false)
  const [showImageMenu, setShowImageMenu] = useState(false)
  const exportRef = useRef(null)
  const imageMenuRef = useRef(null)
  const reportRef = useRef(null)

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
      setActiveTab('original')
      setProgress('')
    } catch (e) {
      setError(e.message)
      setProgress('')
    } finally {
      setLoading(false)
    }
  }

  const tabs = results ? [
    { id: 'original', label: 'Original', icon: <ImageIcon size={16} /> },
    { id: 'attention', label: 'Attention', icon: <Flame size={16} /> },
    { id: 'scanpath', label: 'Scanpath', icon: <Eye size={16} /> },
    { id: 'scroll', label: 'Scroll Depth', icon: <MousePointer2 size={16} /> },
    ...(results.mouse_movement ? [{ id: 'mouse', label: 'Mouse Movement', icon: <Mouse size={16} /> }] : []),
    { id: 'accessibility', label: 'Accessibility', icon: <Accessibility size={16} /> },
    ...(results.ux_overview ? [{ id: 'ux_overview', label: 'UX Overview', icon: <Sparkles size={16} /> }] : []),
  ] : []

  const tabImages = {
    original: results?.original,
    attention: results?.attention,
    scanpath: results?.scanpath,
    scroll: results?.scroll_depth,
    mouse: results?.mouse_movement,
  }

  // Keyboard shortcuts
  const handleKeyDown = useCallback((e) => {
    if (!results) return
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return
    const idx = parseInt(e.key) - 1
    if (idx >= 0 && idx < tabs.length) {
      setActiveTab(tabs[idx].id)
    }
  }, [results, tabs])

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  // Close dropdowns on outside click
  useEffect(() => {
    const handleClick = (e) => {
      if (exportRef.current && !exportRef.current.contains(e.target)) setShowExportMenu(false)
      if (imageMenuRef.current && !imageMenuRef.current.contains(e.target)) setShowImageMenu(false)
    }
    document.addEventListener('click', handleClick)
    return () => document.removeEventListener('click', handleClick)
  }, [])

  // ATF thresholds based on device type
  const deviceType = results?.device_type || 'desktop'
  const atfTarget = deviceType === 'mobile' ? 8 : 15
  const atfPassThreshold = deviceType === 'mobile' ? 8 : 15
  const atfWarnThreshold = deviceType === 'mobile' ? 4 : 8

  const getScoreStatus = (value, passThreshold, warnThreshold) => {
    if (value >= passThreshold) return 'pass'
    if (value >= warnThreshold) return 'warning'
    return 'fail'
  }

  const stats = results ? [
    {
      label: 'Focus Score',
      value: results.focus_score?.toFixed(1),
      suffix: '%',
      target: '>50%',
      desc: 'Attention concentration',
      status: getScoreStatus(results.focus_score, 50, 35)
    },
    {
      label: 'Clarity Score',
      value: results.clarity_score?.toFixed(1),
      suffix: '%',
      target: '>60%',
      desc: 'Visual clutter index',
      status: getScoreStatus(results.clarity_score, 60, 45)
    },
    {
      label: 'Above Fold',
      value: results.above_fold_analysis?.above_fold_attention_pct?.toFixed(1),
      suffix: '%',
      target: `>${atfTarget}%`,
      desc: 'Initial engagement',
      status: getScoreStatus(results.above_fold_analysis?.above_fold_attention_pct, atfPassThreshold, atfWarnThreshold)
    },
    {
      label: 'Attention Areas',
      value: results.boxes?.length,
      suffix: '',
      target: null,
      desc: 'Distinct elements',
      status: 'neutral'
    }
  ] : []

  // Download helpers
  const downloadCurrentImage = () => {
    const imgSrc = tabImages[activeTab]
    if (!imgSrc) return
    const link = document.createElement('a')
    link.download = `analysis-${activeTab}.png`
    link.href = imgSrc
    link.click()
    setShowImageMenu(false)
  }

  const downloadAllImages = () => {
    const entries = Object.entries(tabImages).filter(([, src]) => src)
    entries.forEach(([name, src], i) => {
      setTimeout(() => {
        const link = document.createElement('a')
        link.download = `analysis-${name}.png`
        link.href = src
        link.click()
      }, i * 300)
    })
    setShowImageMenu(false)
  }

  const downloadReportAs = async (format) => {
    const rawMd = results?.accessibility_report || ''
    setShowExportMenu(false)

    if (format === 'md') {
      const blob = new Blob([rawMd], { type: 'text/markdown' })
      triggerBlobDownload(blob, 'accessibility-report.md')
    } else if (format === 'html') {
      const htmlBody = marked.parse(rawMd)
      const htmlContent = `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>Accessibility Report</title><style>body{font-family:Inter,system-ui,sans-serif;max-width:800px;margin:2rem auto;padding:0 2rem;color:#1a1a1a;line-height:1.6}h1,h2,h3{margin-top:2em}table{width:100%;border-collapse:collapse;margin:1rem 0}th,td{padding:.75rem 1rem;border:1px solid #e5e5e5;text-align:left}th{background:#f5f5f5;font-weight:600}hr{border:none;border-top:1px solid #e5e5e5;margin:2rem 0}code{background:#f5f5f5;padding:.15rem .4rem;border-radius:4px;font-size:.9em}ul{padding-left:1.5rem}</style></head><body>${htmlBody}</body></html>`
      const blob = new Blob([htmlContent], { type: 'text/html' })
      triggerBlobDownload(blob, 'accessibility-report.html')
    } else if (format === 'pdf') {
      if (!reportRef.current) return
      try {
        const canvas = await html2canvas(reportRef.current, { scale: 2, useCORS: true })
        const imgData = canvas.toDataURL('image/png')
        const pdf = new jsPDF('p', 'mm', 'a4')
        const pdfWidth = pdf.internal.pageSize.getWidth()
        const pdfHeight = (canvas.height * pdfWidth) / canvas.width
        let heightLeft = pdfHeight
        let position = 0

        pdf.addImage(imgData, 'PNG', 0, position, pdfWidth, pdfHeight)
        heightLeft -= pdf.internal.pageSize.getHeight()

        while (heightLeft > 0) {
          position -= pdf.internal.pageSize.getHeight()
          pdf.addPage()
          pdf.addImage(imgData, 'PNG', 0, position, pdfWidth, pdfHeight)
          heightLeft -= pdf.internal.pageSize.getHeight()
        }

        pdf.save('accessibility-report.pdf')
      } catch (err) {
        console.error('PDF generation failed:', err)
        // Fallback: open print dialog
        const printWin = window.open('', '_blank')
        const htmlBody = marked.parse(rawMd)
        printWin.document.write(`<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Report</title><style>body{font-family:system-ui,sans-serif;max-width:800px;margin:2rem auto;padding:0 2rem;line-height:1.6}table{width:100%;border-collapse:collapse}th,td{padding:.5rem;border:1px solid #ddd}th{background:#f5f5f5}</style></head><body>${htmlBody}</body></html>`)
        printWin.document.close()
        setTimeout(() => printWin.print(), 500)
      }
    }
  }

  const triggerBlobDownload = (blob, filename) => {
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    link.click()
    URL.revokeObjectURL(url)
  }

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
        {!results && (
          <div className="wizard-layout">
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

            <div className="wizard-content">
              {error && <div className="error-banner"><Flame size={18} /> {error}</div>}

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

        {/* Results View */}
        {results && (
          <div className="results-panel fade-in">
            {/* Header with actions */}
            <div className="results-header">
              <div className="results-header-left">
                <button className="btn btn-ghost" onClick={() => { setResults(null); setStep(1) }}>
                  <ArrowLeft size={16} /> New Analysis
                </button>
                <h2 className="results-title">Analysis Complete</h2>
              </div>
              <div className="results-header-right">
                <div className="dropdown" ref={imageMenuRef}>
                  <button className="btn btn-outline btn-sm" onClick={(e) => { e.stopPropagation(); setShowImageMenu(!showImageMenu); setShowExportMenu(false) }}>
                    <Download size={14} /> Images <ChevronDown size={12} />
                  </button>
                  {showImageMenu && (
                    <div className="dropdown-menu">
                      <button className="dropdown-item" onClick={downloadCurrentImage}>
                        <ImageIcon size={14} /> Current Tab PNG
                      </button>
                      <button className="dropdown-item" onClick={downloadAllImages}>
                        <Download size={14} /> All Charts
                      </button>
                    </div>
                  )}
                </div>
                <div className="dropdown" ref={exportRef}>
                  <button className="btn btn-primary btn-sm" onClick={(e) => { e.stopPropagation(); setShowExportMenu(!showExportMenu); setShowImageMenu(false) }}>
                    <FileText size={14} /> Export Report <ChevronDown size={12} />
                  </button>
                  {showExportMenu && (
                    <div className="dropdown-menu">
                      <button className="dropdown-item" onClick={() => downloadReportAs('md')}>
                        Markdown (.md)
                      </button>
                      <button className="dropdown-item" onClick={() => downloadReportAs('html')}>
                        HTML (.html)
                      </button>
                      <button className="dropdown-item" onClick={() => downloadReportAs('pdf')}>
                        PDF (.pdf)
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Metrics Dashboard */}
            <div className="stats-grid">
              {stats.map(s => (
                <div key={s.label} className={`stat-card stat-${s.status}`}>
                  <div className="stat-label">{s.label}</div>
                  <div className="stat-row">
                    <span className={`stat-value`}>{s.value}{s.suffix}</span>
                    {s.target && <span className={`stat-target stat-target-${s.status}`}>{`Target: ${s.target}`}</span>}
                  </div>
                  <div className="stat-desc">{s.desc}</div>
                </div>
              ))}
            </div>

            {/* Tab Navigation */}
            <div className="tabs-bar">
              <div className="tabs-list-pill">
                {tabs.map((tab, i) => (
                  <button
                    key={tab.id}
                    className={`tab-pill ${activeTab === tab.id ? 'active' : ''}`}
                    onClick={() => setActiveTab(tab.id)}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
              <span className="tab-hint">Press 1-{tabs.length} to switch tabs</span>
            </div>

            {/* Tab Content */}
            <div className="tab-content">
              {/* Image tabs */}
              {tabImages[activeTab] && (
                <div className="image-viewer">
                  <img src={tabImages[activeTab]} alt={activeTab} />
                </div>
              )}

              {/* Accessibility Report with TOC */}
              {activeTab === 'accessibility' && (
                <div className="report-layout">
                  <ReportTOC html={marked.parse(results.accessibility_report || '')} />
                  <div
                    ref={reportRef}
                    className="report-markdown prose"
                    dangerouslySetInnerHTML={{ __html: processStatusBadges(marked.parse(results.accessibility_report || '')) }}
                  />
                </div>
              )}

              {/* UX Overview with TOC */}
              {activeTab === 'ux_overview' && results.ux_overview && (
                <div className="report-layout">
                  <ReportTOC html={marked.parse(results.ux_overview)} />
                  <div
                    className="report-markdown prose"
                    dangerouslySetInnerHTML={{ __html: processStatusBadges(marked.parse(results.ux_overview)) }}
                  />
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}


function ReportTOC({ html }) {
  const [activeId, setActiveId] = useState('')
  const [headings, setHeadings] = useState([])

  useEffect(() => {
    const parser = new DOMParser()
    const doc = parser.parseFromString(html, 'text/html')
    const headers = doc.querySelectorAll('h2, h3')
    const items = Array.from(headers).map((h, i) => {
      const id = `section-${i}`
      const realEl = document.getElementById(id)
      return {
        id,
        text: h.textContent,
        level: h.tagName.toLowerCase(),
      }
    })
    setHeadings(items)

    // Inject IDs into the rendered DOM
    setTimeout(() => {
      const container = document.querySelector('.report-markdown')
      if (!container) return
      const rendered = container.querySelectorAll('h2, h3')
      rendered.forEach((h, i) => {
        h.id = `section-${i}`
      })

      // Scroll spy
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            setActiveId(entry.target.id)
          }
        })
      }, { rootMargin: '-10% 0px -80% 0px' })

      rendered.forEach(h => observer.observe(h))
      return () => rendered.forEach(h => observer.unobserve(h))
    }, 100)
  }, [html])

  if (headings.length === 0) return null

  return (
    <aside className="report-toc">
      <div className="toc-sticky">
        <h4 className="toc-title">On This Page</h4>
        <nav className="toc-nav">
          {headings.map(h => (
            <a
              key={h.id}
              href={`#${h.id}`}
              className={`toc-link ${h.level} ${activeId === h.id ? 'active' : ''}`}
              onClick={(e) => {
                e.preventDefault()
                document.getElementById(h.id)?.scrollIntoView({ behavior: 'smooth' })
              }}
            >
              {h.text}
            </a>
          ))}
        </nav>
      </div>
    </aside>
  )
}


function processStatusBadges(html) {
  return html
    .replace(/\[PASS\]/gi, '<span class="status-badge status-pass">&#10003; PASS</span>')
    .replace(/\[WARNING\]/gi, '<span class="status-badge status-warning">&#9888; WARNING</span>')
    .replace(/\[FAIL\]/gi, '<span class="status-badge status-fail">&#10007; FAIL</span>')
}


export default App
