import { useState, useEffect } from 'react'
import './App.css'
import MapVisualization from './components/MapVisualization'
import StatsDashboard from './components/StatsDashboard'
import DemographicBreakdown from './components/DemographicBreakdown'
import SampleRatings from './components/SampleRatings'
import FilteredView from './components/FilteredView'
import GRPODashboard from './components/GRPODashboard'

function App() {
  const [mode, setMode] = useState('grpo') // 'simulation' or 'grpo'
  const [data, setData] = useState(null)
  const [grpoData, setGrpoData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [dataPath, setDataPath] = useState('viz_data')
  const [grpoPath, setGrpoPath] = useState('grpo_data')

  useEffect(() => {
    if (mode === 'simulation') {
      loadSimulationData()
    } else {
      loadGRPOData()
    }
  }, [mode, dataPath, grpoPath])

  const loadSimulationData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [overall, zipcode, state, demographics, samples] = await Promise.all([
        fetch(`/${dataPath}/overall.json`).then(r => r.json()),
        fetch(`/${dataPath}/by_zipcode.json`).then(r => r.json()),
        fetch(`/${dataPath}/by_state.json`).then(r => r.json()),
        fetch(`/${dataPath}/by_demographics.json`).then(r => r.json()),
        fetch(`/${dataPath}/sample_ratings.json`).then(r => r.json()),
      ])
      
      let rawRatings = null
      try {
        rawRatings = await fetch(`/${dataPath}/raw_ratings.json`).then(r => r.json())
      } catch (e) {
        console.log('Raw ratings not available')
      }
      
      setData({ overall, zipcode, state, demographics, samples, rawRatings })
    } catch (err) {
      setError(`Failed to load simulation data from ${dataPath}.`)
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const loadGRPOData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [evalResults, evalVotes, personaSets] = await Promise.all([
        fetch(`/${grpoPath}/eval_results.json`).then(r => r.json()),
        fetch(`/${grpoPath}/eval_votes.json`).then(r => r.json()).catch(() => null),
        fetch(`/${grpoPath}/persona_sets.json`).then(r => r.json()).catch(() => null),
      ])
      
      setGrpoData({ evalResults, evalVotes, personaSets })
    } catch (err) {
      setError(`Failed to load GRPO data from ${grpoPath}. Copy your run's eval_results.json, eval_votes.json, and persona_sets.json to public/${grpoPath}/`)
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="app">
        <div className="loading">Loading data...</div>
      </div>
    )
  }

  // Mode selector header
  const ModeHeader = () => (
    <div className="mode-header">
      <div className="mode-tabs">
        <button 
          className={`mode-tab ${mode === 'grpo' ? 'active' : ''}`}
          onClick={() => setMode('grpo')}
        >
          🎯 GRPO Training
        </button>
        <button 
          className={`mode-tab ${mode === 'simulation' ? 'active' : ''}`}
          onClick={() => setMode('simulation')}
        >
          📊 Persona Simulation
        </button>
      </div>
      <div className="data-path-selector">
        <label>Data path:</label>
        <input 
          type="text" 
          value={mode === 'grpo' ? grpoPath : dataPath} 
          onChange={(e) => mode === 'grpo' ? setGrpoPath(e.target.value) : setDataPath(e.target.value)}
          placeholder={mode === 'grpo' ? 'grpo_data' : 'viz_data'}
        />
      </div>
    </div>
  )

  if (error) {
    return (
      <div className="app">
        <ModeHeader />
        <div className="error">
          <h2>Error</h2>
          <p>{error}</p>
          <p className="hint">
            {mode === 'grpo' ? (
              <>Copy your run files to <code>public/{grpoPath}/</code>:<br/>
              <code>cp runs/YOUR_RUN/eval_results.json public/{grpoPath}/</code><br/>
              <code>cp runs/YOUR_RUN/eval_votes.json public/{grpoPath}/</code><br/>
              <code>cp runs/YOUR_RUN/persona_sets.json public/{grpoPath}/</code></>
            ) : (
              <>Run <code>aggregate_results.py</code> first to generate visualization data.</>
            )}
          </p>
        </div>
      </div>
    )
  }

  // GRPO Mode
  if (mode === 'grpo' && grpoData) {
    return (
      <div className="app">
        <ModeHeader />
        <header className="masthead">
          <h1>GRPO Training Dashboard</h1>
          <p className="subtitle">
            Training a model to write tweets that resonate with a target demographic, evaluated against GPT-4.1
          </p>
        </header>
        <GRPODashboard 
          evalResults={grpoData.evalResults} 
          evalVotes={grpoData.evalVotes}
          personaSets={grpoData.personaSets}
        />
      </div>
    )
  }

  // Simulation Mode (original Day 18)
  if (mode === 'simulation' && data) {
    const content = data.overall?.config?.content || "Content not available"
    const totalPersonas = data.overall?.n?.toLocaleString() || "—"

    return (
      <div className="app">
        <ModeHeader />
        <header className="masthead">
          <div className="masthead-top">
            <span className="masthead-section">Opinion Research</span>
            <span className="masthead-date">{new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</span>
          </div>
          <h1>How America Reacts</h1>
          <p className="subtitle">We asked {totalPersonas} AI personas—each embodying real U.S. demographics—to react to a single social media post. Here's what they said.</p>
        </header>

        <section className="content-showcase">
          <div className="content-label">The Post</div>
          <blockquote className="content-quote">
            "{content}"
          </blockquote>
          <div className="content-meta">
            Personas were asked to rate this content on a scale of 1-10 for likeability and emotional activation, responding as their assigned demographic identity.
          </div>
        </section>

        <StatsDashboard overall={data.overall} />

        <section className="map-section">
          <h2>Geographic Distribution</h2>
          <MapVisualization zipcodeData={data.zipcode} stateData={data.state} />
        </section>

        {data.rawRatings && (
          <section className="filter-section">
            <h2>Explore by Demographics</h2>
            <p className="section-subtitle">Filter personas and see how different groups react</p>
            <FilteredView rawRatings={data.rawRatings} />
          </section>
        )}

        <section className="demographics-section">
          <h2>Demographic Breakdown</h2>
          <DemographicBreakdown demographics={data.demographics} />
        </section>

        <section className="samples-section">
          <h2>Sample Responses</h2>
          <SampleRatings samples={data.samples} />
        </section>
      </div>
    )
  }

  return (
    <div className="app">
      <ModeHeader />
      <div className="error">No data loaded</div>
    </div>
  )
}

export default App
