import { useEffect, useMemo, useState } from 'react'
import {
  Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
  BarChart, Bar
} from 'recharts'
import {
  Activity, ArrowDownRight, ArrowUpRight, BatteryCharging, CalendarDays,
  ChevronDown, CircleHelp, Cpu, Download, Gauge, LayoutDashboard, Leaf,
  Menu, Settings2, ShieldCheck, Sparkles, Zap, Save, RefreshCw, Database,
  Bell, Palette, FlaskConical, SlidersHorizontal, Table2, BarChart2
} from 'lucide-react'

const API = 'http://127.0.0.1:8000'
const fallbackTrend = [
  { month: 'Jan 2025', energy_kwh: 1160 }, { month: 'Feb 2025', energy_kwh: 1085 },
  { month: 'Mar 2025', energy_kwh: 1230 }, { month: 'Apr 2025', energy_kwh: 1195 },
  { month: 'May 2025', energy_kwh: 1300 }, { month: 'Jun 2025', energy_kwh: 1210 },
  { month: 'Jul 2025', energy_kwh: 1275 }, { month: 'Aug 2025', energy_kwh: 1350 },
]
const currencyLocales = { INR: 'en-IN', USD: 'en-US', EUR: 'de-DE', GBP: 'en-GB' }
const money = (n, cur = 'INR') => {
  try {
    return new Intl.NumberFormat(currencyLocales[cur] || 'en-IN', { style: 'currency', currency: cur, maximumFractionDigits: 0 }).format(n ?? 0)
  } catch {
    return `₹${Math.round(n ?? 0)}`
  }
}
const number = (n, digits = 1) => new Intl.NumberFormat('en-IN', { maximumFractionDigits: digits }).format(n ?? 0)

function StatCard({ icon: Icon, label, value, unit, change, positive, accent }) {
  return <div className="stat-card">
    <div className="stat-top"><span className={`stat-icon ${accent}`}><Icon size={18}/></span><span className="stat-label">{label}</span><button className="icon-button" aria-label={`About ${label}`}><CircleHelp size={15}/></button></div>
    <div className="stat-value">{value}<span>{unit}</span></div>
    <div className={`stat-change ${positive ? 'good' : 'muted'}`}>{positive ? <ArrowDownRight size={15}/> : <ArrowUpRight size={15}/>} {change}</div>
  </div>
}

export default function App() {
  const [summary, setSummary] = useState(null)
  const [apiState, setApiState] = useState('connecting')
  const [computers, setComputers] = useState(28)
  const [hours, setHours] = useState(6)
  const [watts, setWatts] = useState(130)
  const [tariff, setTariff] = useState(8)
  const [activeCurrency, setActiveCurrency] = useState('INR')
  const [prediction, setPrediction] = useState(null)
  const [predicting, setPredicting] = useState(false)
  const [error, setError] = useState('')
  const [range, setRange] = useState('8 months')
  const [activeSection, setActiveSection] = useState('overview')
  // Historical Analysis state
  const [histSearch, setHistSearch] = useState('')
  const [histSort, setHistSort] = useState({ key: 'date', dir: 'desc' })
  const [histPage, setHistPage] = useState(1)
  const HIST_PAGE_SIZE = 10
  // Settings state
  const [settingTariff, setSettingTariff] = useState(8)
  const [settingCurrency, setSettingCurrency] = useState('INR')
  const [settingNotify, setSettingNotify] = useState(true)
  const [settingSummary, setSettingSummary] = useState(true)
  const [settingDarkSidebar, setSettingDarkSidebar] = useState(true)
  const [settingsSaved, setSettingsSaved] = useState(false)

  useEffect(() => {
    fetch(`${API}/api/summary`)
      .then(r => { if (!r.ok) throw new Error('API unavailable'); return r.json() })
      .then(data => { setSummary(data); setApiState('online') })
      .catch(() => setApiState('offline'))
  }, [])

  const trend = useMemo(() => {
    const data = summary?.monthly_trend?.length ? summary.monthly_trend : fallbackTrend
    return range === '4 months' ? data.slice(-4) : data
  }, [summary, range])

  async function runPrediction(e) {
    e.preventDefault()
    setPredicting(true); setError(''); setPrediction(null)
    try {
      const response = await fetch(`${API}/api/predict`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          computers_used: Number(computers),
          operating_hours: Number(hours),
          avg_power_watts: Number(watts),
          tariff_per_kwh: Number(tariff)
        })
      })
      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        throw new Error(data.detail || 'Prediction request failed')
      }
      setPrediction(await response.json())
    } catch {
      // Transparent offline estimate for frontend preview; never label it as an ML result.
      const estimate = Number(computers) * Number(hours) * Number(watts) / 1000
      setPrediction({ predicted_kwh: Number(estimate.toFixed(2)), estimated_cost: Number((estimate * Number(tariff)).toFixed(2)), demo: true })
      setError('Backend is offline. Showing a basic formula estimate, not a machine-learning prediction.')
    } finally { setPredicting(false) }
  }

  const total = summary?.total_kwh ?? 8624.7
  const average = summary?.average_daily_kwh ?? 35.9
  const days = summary?.days_recorded ?? 240
  const peak = summary?.peak_daily_kwh ?? 58.4

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><Zap size={21} fill="currentColor"/></div><div><div className="brand-name">lab<span>energy</span></div><div className="brand-sub">ENERGY INTELLIGENCE</div></div></div>
      <div className="workspace"><div className="workspace-icon"><Cpu size={18}/></div><div className="workspace-text"><strong>Computer Lab 01</strong><span>Campus facilities</span></div><ChevronDown size={15}/></div>
      <div className="nav-caption">WORKSPACE</div>
      <nav className="nav-links">
        <a className={`nav-link ${activeSection === 'overview' ? 'active' : ''}`} href="#overview" onClick={() => setActiveSection('overview')}><LayoutDashboard size={17}/> Overview {activeSection === 'overview' && <span className="nav-dot"/>}</a>
        <a className={`nav-link ${activeSection === 'prediction' ? 'active' : ''}`} href="#prediction" onClick={() => setActiveSection('prediction')}><Sparkles size={17}/> Predictions <span className="nav-tag">ML</span></a>
        <a className={`nav-link ${activeSection === 'trends' ? 'active' : ''}`} href="#trends" onClick={() => setActiveSection('trends')}><Activity size={17}/> Usage analytics</a>
        <a className={`nav-link ${activeSection === 'efficiency' ? 'active' : ''}`} href="#efficiency" onClick={() => setActiveSection('efficiency')}><Leaf size={17}/> Efficiency insights</a>
      </nav>
      <div className="nav-caption data-caption">DATA & SYSTEM</div>
      <nav className="nav-links">
        <a className={`nav-link ${activeSection === 'historical' ? 'active' : ''}`} href="#historical" onClick={() => setActiveSection('historical')}><CalendarDays size={17}/> Historical data {activeSection === 'historical' && <span className="nav-dot"/>}</a>
        <a className={`nav-link ${activeSection === 'settings' ? 'active' : ''}`} href="#settings" onClick={() => setActiveSection('settings')}><Settings2 size={17}/> Settings {activeSection === 'settings' && <span className="nav-dot"/>}</a>
      </nav>
      <div className="sidebar-bottom">
        <div className="help-card"><div className="help-icon"><CircleHelp size={17}/></div><strong>Need a hand?</strong><p>Explore how the energy forecast works.</p><a href="#prediction">View prediction guide <span>↗</span></a></div>
        <div className="profile"><div className="avatar">CL</div><div className="profile-info"><strong>Lab Administrator</strong><span>Demo workspace</span></div><ChevronDown size={15}/></div>
      </div>
    </aside>

    <main className="main-content" id="overview">
      <header className="topbar"><div className="breadcrumbs"><span>Workspace</span><span className="crumb-slash">/</span><strong>Overview</strong></div><div className="top-actions"><span className={`connection ${apiState}`}><span className="connection-dot"/>{apiState === 'online' ? 'API connected' : apiState === 'offline' ? 'Demo mode' : 'Connecting'}</span><a href="#historical" className="date-button" style={{textDecoration:'none'}} onClick={() => setActiveSection('historical')}><CalendarDays size={15}/> Historical overview <ChevronDown size={14}/></a><div className="avatar small-avatar">CL</div></div></header>
      <div className="page-wrap">
        <section className="page-heading"><div><div className="eyebrow"><span className="eyebrow-line"/> ENERGY OVERVIEW</div><h1>Good morning, <span>Administrator.</span></h1><p className="subtitle">Here’s how your computer lab is using energy over time.</p></div><button className="export-button" onClick={() => {
          const rows = [['Metric','Value'],['Total historical energy (kWh)',total],['Average daily energy (kWh)',average],['Peak daily energy (kWh)',peak],['Days recorded',days]]
          const blob = new Blob([rows.map(r => r.join(',')).join('\n')], { type: 'text/csv' })
          const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = 'lab-energy-summary.csv'; a.click(); URL.revokeObjectURL(url)
        }}><Download size={16}/> Export report</button></section>

        <section className="notice"><div className="notice-icon"><ShieldCheck size={18}/></div><div><strong>Historical analysis mode</strong><p>This dashboard uses saved records and model estimates. No live sensors or real-time readings are connected.</p></div><span className="notice-pill">OFFLINE DATA</span></section>

        <section className="stats-grid">
          <StatCard icon={Zap} label="Total energy recorded" value={number(total)} unit=" kWh" change="Across available dataset" positive accent="purple"/>
          <StatCard icon={Gauge} label="Average daily usage" value={number(average, 2)} unit=" kWh" change={`${days} historical records`} positive accent="blue"/>
          <StatCard icon={BatteryCharging} label="Peak daily usage" value={number(peak, 1)} unit=" kWh" change="Highest single-day record" positive={false} accent="orange"/>
          <StatCard icon={Leaf} label="Estimated total cost" value={money(total * tariff, activeCurrency)} unit="" change={`At ${activeCurrency === 'USD' ? '$' : activeCurrency === 'EUR' ? '€' : activeCurrency === 'GBP' ? '£' : '₹'}${tariff}/kWh illustrative tariff`} positive accent="green"/>
        </section>

        <section className="content-grid">
          <div className="panel trend-panel" id="trends">
            <div className="panel-heading"><div><div className="panel-title">Energy consumption</div><div className="panel-subtitle">Historical usage across the computer lab</div></div><select value={range} onChange={e => setRange(e.target.value)} className="select-control"><option>8 months</option><option>4 months</option></select></div>
            <div className="chart-legend"><span className="legend-dot"/><span>Energy consumed</span><span className="legend-unit">kWh / month</span></div>
            <div className="trend-chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={trend} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}><defs><linearGradient id="energyFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#7c6cf2" stopOpacity={0.28}/><stop offset="100%" stopColor="#7c6cf2" stopOpacity={0.01}/></linearGradient></defs><CartesianGrid stroke="#e9eaf2" strokeDasharray="3 5" vertical={false}/><XAxis dataKey="month" tick={{ fill: '#85889b', fontSize: 11 }} tickLine={false} axisLine={false} tickFormatter={v => v.split(' ')[0]}/><YAxis tick={{ fill: '#85889b', fontSize: 11 }} tickLine={false} axisLine={false}/><Tooltip contentStyle={{ borderRadius: 12, border: '1px solid #e8e9f1', boxShadow: '0 8px 24px #20204412' }} formatter={v => [`${number(v)} kWh`, 'Energy']}/><Area type="monotone" dataKey="energy_kwh" stroke="#7867ef" strokeWidth={2.5} fill="url(#energyFill)" activeDot={{ r: 5, strokeWidth: 0 }}/></AreaChart></ResponsiveContainer></div>
            <div className="chart-footer"><span><span className="footer-check">✓</span> Historical data loaded</span><span>Source: CSV records</span></div>
          </div>

          <div className="panel prediction-panel" id="prediction">
            <div className="panel-heading"><div><div className="panel-title">Energy predictor</div><div className="panel-subtitle">Estimate a lab session’s energy use</div></div><span className="ml-chip"><Sparkles size={12}/> ML MODEL</span></div>
            <form onSubmit={runPrediction} className="predict-form">
              <label>Computers in use <span>{computers} units</span><input type="range" min="1" max="100" value={computers} onChange={e => setComputers(e.target.value)} /><div className="range-labels"><span>1 computer</span><span>100 computers</span></div></label>
              <div className="form-row"><label>Operating hours<input type="number" min="0.1" max="24" step="0.5" value={hours} onChange={e => setHours(e.target.value)} required/></label><label>Power / computer<input type="number" min="1" max="2000" value={watts} onChange={e => setWatts(e.target.value)} required/><small>Watts (W)</small></label></div>
              <label>Electricity tariff <div className="tariff-input"><span>{activeCurrency === 'USD' ? '$' : activeCurrency === 'EUR' ? '€' : activeCurrency === 'GBP' ? '£' : '₹'}</span><input type="number" min="0" max="1000" step="0.5" value={tariff} onChange={e => setTariff(e.target.value)} required/><span>/ kWh</span></div></label>
              <button className="predict-button" type="submit" disabled={predicting}><Sparkles size={16}/>{predicting ? 'Calculating…' : 'Predict energy use'}<span>→</span></button>
            </form>
            {error && <div className="inline-warning">{error}</div>}
            {prediction ? <div className="prediction-result"><div><span className="result-label">{prediction.demo ? 'Formula-based estimate' : 'Predicted energy'}</span><strong>{number(prediction.predicted_kwh, 2)} <small>kWh</small></strong></div><div className="result-cost"><span>Estimated session cost</span><strong>{money(prediction.estimated_cost, activeCurrency)}</strong></div></div> : <div className="predict-hint"><div className="hint-spark"><Sparkles size={16}/></div><span>Adjust the inputs and run a prediction to estimate energy and cost for a lab session.</span></div>}
          </div>
        </section>

        <section className="bottom-grid">
          <div className="panel insights-panel" id="efficiency"><div className="panel-heading"><div><div className="panel-title">Efficiency insights</div><div className="panel-subtitle">Practical ways to reduce energy use</div></div><span className="insight-count">03 TIPS</span></div>
            <div className="insight-row"><div className="insight-icon lavender"><Cpu size={17}/></div><div className="insight-copy"><strong>Review idle computer time</strong><p>Track sessions where computers remain on without active use.</p></div><span className="insight-arrow">↗</span></div>
            <div className="insight-row"><div className="insight-icon mint"><Leaf size={17}/></div><div className="insight-copy"><strong>Compare operating schedules</strong><p>Use historical patterns to identify lower-use time windows.</p></div><span className="insight-arrow">↗</span></div>
            <div className="insight-row"><div className="insight-icon peach"><Zap size={17}/></div><div className="insight-copy"><strong>Model a shorter session</strong><p>Compare predicted energy at different operating durations.</p></div><span className="insight-arrow">↗</span></div>
          </div>
          <div className="panel dataset-panel" id="dataset"><div className="panel-heading"><div><div className="panel-title">Dataset health</div><div className="panel-subtitle">Input coverage and model status</div></div><span className={`status-badge ${apiState === 'online' ? 'status-ok' : 'status-demo'}`}>{apiState === 'online' ? 'Connected' : 'Demo'}</span></div>
            <div className="dataset-stat"><div className="dataset-stat-icon"><CalendarDays size={17}/></div><div><strong>{days} days</strong><span>Historical records</span></div><span className="dataset-check">✓</span></div>
            <div className="dataset-stat"><div className="dataset-stat-icon"><Activity size={17}/></div><div><strong>{summary?.metrics?.features?.length ?? 3} model features</strong><span>Computers, hours, power</span></div><span className="dataset-check">✓</span></div>
            <div className="dataset-stat"><div className="dataset-stat-icon"><ShieldCheck size={17}/></div><div><strong>Sample dataset</strong><span>Replace with verified readings</span></div><span className="dataset-check warning-check">!</span></div>
            <div className="dataset-footnote">Model quality depends on the accuracy and representativeness of the input data.</div>
          </div>
        </section>

        {/* ── Historical Analysis ── */}
        <section id="historical" style={{scrollMarginTop:'20px'}}>
          <div className="section-header">
            <div>
              <div className="eyebrow"><span className="eyebrow-line"/> HISTORICAL ANALYSIS</div>
              <h2 className="section-title">Historical data</h2>
              <p className="subtitle">Browse, filter and export all recorded lab sessions.</p>
            </div>
            <button className="export-button" onClick={()=>{
              const rows = [['Date','Computers Used','Operating Hours','Avg Power (W)','Energy (kWh)']]
              const data = summary?.recent_days ?? []
              ;(summary?.monthly_trend ?? fallbackTrend).forEach((r,i)=>rows.push([r.month, '-', '-', '-', r.energy_kwh]))
              const blob = new Blob([rows.map(r=>r.join(',')).join('\n')],{type:'text/csv'})
              const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='lab-historical-data.csv';a.click();URL.revokeObjectURL(url)
            }}><Download size={16}/> Export CSV</button>
          </div>

          <div className="panel hist-panel">
            <div className="panel-heading hist-toolbar">
              <div className="hist-search-wrap">
                <input className="hist-search" placeholder="🔍  Search by month or value…" value={histSearch} onChange={e=>{setHistSearch(e.target.value);setHistPage(1)}}/>
              </div>
              <div className="hist-chart-toggle">
                <span style={{fontSize:10,color:'#9a9db0'}}>Monthly breakdown</span>
              </div>
            </div>

            {/* Bar chart */}
            <div className="hist-chart">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={summary?.monthly_trend?.length ? summary.monthly_trend : fallbackTrend} margin={{top:8,right:8,left:-16,bottom:0}}>
                  <CartesianGrid stroke="#e9eaf2" strokeDasharray="3 5" vertical={false}/>
                  <XAxis dataKey="month" tick={{fill:'#85889b',fontSize:11}} tickLine={false} axisLine={false} tickFormatter={v=>v.split(' ')[0]}/>
                  <YAxis tick={{fill:'#85889b',fontSize:11}} tickLine={false} axisLine={false}/>
                  <Tooltip contentStyle={{borderRadius:12,border:'1px solid #e8e9f1',boxShadow:'0 8px 24px #20204412'}} formatter={v=>[`${number(v)} kWh`,'Energy']}/>
                  <Bar dataKey="energy_kwh" fill="#7867ef" radius={[6,6,0,0]}/>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Table */}
            <div className="hist-table-wrap">
              <table className="hist-table">
                <thead>
                  <tr>
                    {[['month','Month / Period'],['energy_kwh','Energy (kWh)']].map(([k,label])=>(
                      <th key={k} onClick={()=>setHistSort(s=>({key:k,dir:s.key===k&&s.dir==='asc'?'desc':'asc'}))} className="sortable-th">
                        {label} {histSort.key===k?(histSort.dir==='asc'?'↑':'↓'):'↕'}
                      </th>
                    ))}
                    <th>Share of Total</th>
                  </tr>
                </thead>
                <tbody>
                  {(()=>{
                    const raw = summary?.monthly_trend?.length ? summary.monthly_trend : fallbackTrend
                    const totalKwh = raw.reduce((a,r)=>a+r.energy_kwh,0)
                    const filtered = raw.filter(r=>r.month.toLowerCase().includes(histSearch.toLowerCase())||(String(r.energy_kwh).includes(histSearch)))
                    const sorted = [...filtered].sort((a,b)=>{
                      const av=a[histSort.key],bv=b[histSort.key]
                      return histSort.dir==='asc'?(av>bv?1:-1):(av<bv?1:-1)
                    })
                    const paged = sorted.slice((histPage-1)*HIST_PAGE_SIZE,histPage*HIST_PAGE_SIZE)
                    return paged.length===0
                      ? <tr><td colSpan={3} style={{textAlign:'center',padding:'24px',color:'#aaa',fontSize:12}}>No records match your search.</td></tr>
                      : paged.map((r,i)=>{
                          const pct=(r.energy_kwh/totalKwh*100).toFixed(1)
                          return <tr key={i}>
                            <td><span className="hist-month-badge">{r.month}</span></td>
                            <td><strong>{number(r.energy_kwh,1)}</strong> kWh</td>
                            <td><div className="hist-bar-cell"><div className="hist-bar-track"><div className="hist-bar-fill" style={{width:`${pct}%`}}/></div><span>{pct}%</span></div></td>
                          </tr>
                        })
                  })()}
                </tbody>
              </table>
              <div className="hist-pagination">
                <button className="page-btn" disabled={histPage===1} onClick={()=>setHistPage(p=>p-1)}>← Prev</button>
                <span className="page-info">Page {histPage}</span>
                <button className="page-btn" disabled={(summary?.monthly_trend?.length||fallbackTrend.length)<=histPage*HIST_PAGE_SIZE} onClick={()=>setHistPage(p=>p+1)}>Next →</button>
              </div>
            </div>
          </div>
        </section>

        {/* ── Settings ── */}
        <section id="settings" style={{marginTop:'36px'}}>
          <div className="section-header">
            <div>
              <div className="eyebrow"><span className="eyebrow-line"/> CONFIGURATION</div>
              <h2 className="section-title">Settings</h2>
              <p className="subtitle">Customize how LabEnergy calculates and displays data.</p>
            </div>
            {settingsSaved && <span className="save-toast">✓ Settings saved</span>}
          </div>

          <div className="settings-grid">
            {/* Tariff & Cost */}
            <div className="panel settings-panel">
              <div className="settings-section-head"><SlidersHorizontal size={16}/> Tariff &amp; Cost</div>
              <div className="settings-field">
                <label className="settings-label">Electricity tariff ({settingCurrency === 'USD' ? '$' : settingCurrency === 'EUR' ? '€' : settingCurrency === 'GBP' ? '£' : '₹'} per kWh)</label>
                <p className="settings-desc">Used to calculate estimated session costs across the dashboard.</p>
                <div className="tariff-input settings-input">
                  <span>{settingCurrency === 'USD' ? '$' : settingCurrency === 'EUR' ? '€' : settingCurrency === 'GBP' ? '£' : '₹'}</span>
                  <input type="number" min="0" max="1000" step="0.5" value={settingTariff} onChange={e=>setSettingTariff(e.target.value)}/>
                  <span>/ kWh</span>
                </div>
              </div>
              <div className="settings-field">
                <label className="settings-label">Display currency</label>
                <p className="settings-desc">All monetary values will use this currency symbol.</p>
                <select className="select-control settings-select" value={settingCurrency} onChange={e=>setSettingCurrency(e.target.value)}>
                  <option value="INR">₹ Indian Rupee (INR)</option>
                  <option value="USD">$ US Dollar (USD)</option>
                  <option value="EUR">€ Euro (EUR)</option>
                  <option value="GBP">£ British Pound (GBP)</option>
                </select>
              </div>
            </div>

            {/* Notifications */}
            <div className="panel settings-panel">
              <div className="settings-section-head"><Bell size={16}/> Notifications &amp; Alerts</div>
              <div className="settings-field">
                <div className="settings-toggle-row">
                  <div>
                    <label className="settings-label">High usage alerts</label>
                    <p className="settings-desc">Notify when daily usage exceeds the historical peak.</p>
                  </div>
                  <button className={`toggle-btn${settingNotify?' toggle-on':''}`} onClick={()=>setSettingNotify(v=>!v)} aria-label="Toggle alerts">
                    <span className="toggle-knob"/>
                  </button>
                </div>
              </div>
              <div className="settings-field">
                <div className="settings-toggle-row">
                  <div>
                    <label className="settings-label">Monthly summary</label>
                    <p className="settings-desc">Generate a report at the end of each month.</p>
                  </div>
                  <button className={`toggle-btn${settingSummary?' toggle-on':''}`} onClick={()=>setSettingSummary(v=>!v)} aria-label="Toggle summary">
                    <span className="toggle-knob"/>
                  </button>
                </div>
              </div>
            </div>

            {/* Model info */}
            <div className="panel settings-panel">
              <div className="settings-section-head"><FlaskConical size={16}/> ML Model Info</div>
              <div className="settings-field">
                <label className="settings-label">Model type</label>
                <p className="settings-desc">Random Forest Regressor · 250 estimators</p>
              </div>
              <div className="model-metrics-grid">
                <div className="model-metric"><span>R² Score</span><strong>{summary?.metrics?.r2 ?? '0.923'}</strong></div>
                <div className="model-metric"><span>MAE</span><strong>{summary?.metrics?.mae_kwh ?? '1.7012'} kWh</strong></div>
                <div className="model-metric"><span>Train rows</span><strong>{summary?.metrics?.training_rows ?? 192}</strong></div>
                <div className="model-metric"><span>Test rows</span><strong>{summary?.metrics?.test_rows ?? 48}</strong></div>
              </div>
              <div className="settings-field" style={{marginBottom:0}}>
                <label className="settings-label">Features used</label>
                <div style={{display:'flex',gap:6,flexWrap:'wrap',marginTop:8}}>
                  {(summary?.metrics?.features ?? ['computers_used','operating_hours','avg_power_watts']).map(f=>
                    <span key={f} className="feature-tag">{f}</span>
                  )}
                </div>
              </div>
            </div>

            {/* Data management */}
            <div className="panel settings-panel">
              <div className="settings-section-head"><Database size={16}/> Data Management</div>
              <div className="settings-field">
                <label className="settings-label">Dataset source</label>
                <p className="settings-desc">Currently using the bundled synthetic demonstration dataset. Replace with verified lab readings for production use.</p>
              </div>
              <div className="settings-actions">
                <button className="settings-action-btn" onClick={()=>{
                  const rows=[['date','computers_used','operating_hours','avg_power_watts','energy_kwh']]
                  ;(summary?.monthly_trend??fallbackTrend).forEach(r=>rows.push([r.month,'—','—','—',r.energy_kwh]))
                  const blob=new Blob([rows.map(r=>r.join(',')).join('\n')],{type:'text/csv'})
                  const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='lab-energy-export.csv';a.click();URL.revokeObjectURL(url)
                }}><Download size={14}/> Export all data</button>
                <button className="settings-action-btn danger-btn" onClick={()=>alert('In a production system this would clear cached data. No data was deleted in this demo.')}>
                  <RefreshCw size={14}/> Clear cache
                </button>
              </div>
            </div>
          </div>

          <div className="settings-save-row">
            <button className="predict-button settings-save-btn" onClick={()=>{
              setTariff(Number(settingTariff))
              setActiveCurrency(settingCurrency)
              setSettingsSaved(true)
              setTimeout(()=>setSettingsSaved(false),2500)
            }}><Save size={15}/> Save settings</button>
            <span style={{fontSize:11,color:'#8c8fa4'}}>Changes to tariff and currency are immediately applied across the dashboard and predictions.</span>
          </div>
        </section>

        <footer className="footer"><span>© 2025 LabEnergy · Computer Lab Energy Intelligence</span><span><span className="footer-live-dot"/> Built for historical analysis <span className="footer-divider">·</span> Not a live monitoring system</span></footer>
      </div>
    </main>
  </div>
}
