import { useState, useEffect } from 'react'
import './App.css'
import MapVisualization from './components/MapVisualization'
import StatsDashboard from './components/StatsDashboard'
import DemographicBreakdown from './components/DemographicBreakdown'
import SampleRatings from './components/SampleRatings'
import FilteredView from './components/FilteredView'

function App() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [dataPath, setDataPath] = useState('viz_data')

  useEffect(() => {
    loadData()
  }, [dataPath])

  const loadData = async () => {
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
      
      // Try to load raw ratings (optional, for filtering)
      let rawRatings = null
      try {
        rawRatings = await fetch(`/${dataPath}/raw_ratings.json`).then(r => r.json())
      } catch (e) {
        console.log('Raw ratings not available (run aggregate with --include-raw)')
      }
      
      setData({
        overall,
        zipcode,
        state,
        demographics,
        samples,
        rawRatings,
      })
    } catch (err) {
      setError(`Failed to load data from ${dataPath}. Make sure you've run aggregate_results.py first.`)
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

  if (error || !data) {
    return (
      <div className="app">
        <div className="error">
          <h2>Error</h2>
          <p>{error || 'No data available'}</p>
          <div className="data-path-input">
            <label>Data path:</label>
            <input 
              type="text" 
              value={dataPath} 
              onChange={(e) => setDataPath(e.target.value)}
              placeholder="viz_data"
            />
            <button onClick={loadData}>Load</button>
          </div>
        </div>
      </div>
    )
  }

  const content = data.overall?.config?.content || "Content not available"
  const totalPersonas = data.overall?.n?.toLocaleString() || "—"

  return (
    <div className="app">
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

export default App
