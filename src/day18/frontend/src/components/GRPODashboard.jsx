import { useState, useMemo } from 'react'
import './GRPODashboard.css'
import MapVisualization from './MapVisualization'

function GRPODashboard({ evalResults, evalVotes, personaSets }) {
  const [selectedStep, setSelectedStep] = useState(0)
  const [showTarget, setShowTarget] = useState(true) // toggle target vs general
  
  // Get sorted steps
  const steps = useMemo(() => {
    return Object.keys(evalResults)
      .map(Number)
      .sort((a, b) => a - b)
  }, [evalResults])
  
  // Get step index for slider
  const stepIndex = steps.indexOf(selectedStep)
  const currentStepIndex = stepIndex >= 0 ? stepIndex : 0
  
  // Build persona lookup for demographics
  const personaLookup = useMemo(() => {
    const lookup = {}
    if (personaSets?.eval_target) {
      personaSets.eval_target.forEach(p => { lookup[p.uuid] = { ...p, in_target: true } })
    }
    if (personaSets?.eval_general) {
      personaSets.eval_general.forEach(p => { 
        if (!lookup[p.uuid]) lookup[p.uuid] = { ...p, in_target: false } 
      })
    }
    return lookup
  }, [personaSets])
  
  // Get target demographic description
  const targetDemoDescription = useMemo(() => {
    if (!personaSets?.target_filters) return null
    const filters = personaSets.target_filters
    const parts = []
    if (filters.age) {
      parts.push(`Age ${filters.age.min}-${filters.age.max}`)
    }
    if (filters.education_level) {
      parts.push(`Education: ${filters.education_level.join(', ')}`)
    }
    if (filters.state) {
      parts.push(`States: ${filters.state.join(', ')}`)
    }
    return parts.length > 0 ? parts.join(' • ') : 'Custom demographic filter'
  }, [personaSets])
  
  const currentData = evalResults[steps[currentStepIndex]]
  const currentVotes = evalVotes?.[steps[currentStepIndex]]
  
  // Pick a representative model tweet (first one, or could pick best)
  const displayTweet = currentData?.model_tweets?.[0] || 'No tweet available'
  
  // Compute state-level vote data for map
  const stateData = useMemo(() => {
    if (!currentVotes || !personaLookup) return {}
    
    const votes = showTarget ? currentVotes.target_votes : currentVotes.general_votes
    if (!votes) return {}
    
    const byState = {}
    
    votes.forEach(vote => {
      const persona = personaLookup[vote.uuid]
      if (!persona || !vote.voted_for) return
      
      const state = persona.state
      if (!state) return
      
      if (!byState[state]) {
        byState[state] = { model: 0, gpt: 0, total: 0 }
      }
      byState[state][vote.voted_for]++
      byState[state].total++
    })
    
    // Convert to format MapVisualization expects
    // Map expects: { STATE: { likeability_mean: number, emotional_mean: number, n: number } }
    // We'll map win_rate (0-1) to a 3-8 scale where 5.5 is neutral (50%)
    const mapData = {}
    Object.entries(byState).forEach(([state, votes]) => {
      const winRate = votes.total > 0 ? votes.model / votes.total : 0.5
      // Map 0-1 win rate to 3-8 scale (centered at 5.5 for 50%)
      const scaledValue = 3 + (winRate * 5) // 0% -> 3, 50% -> 5.5, 100% -> 8
      mapData[state] = {
        likeability_mean: scaledValue,
        emotional_mean: scaledValue, // Same for both metrics
        win_rate: winRate,
        model_votes: votes.model,
        gpt_votes: votes.gpt,
        n: votes.total,
      }
    })
    
    return mapData
  }, [currentVotes, personaLookup, showTarget])
  
  // Overall win rate for current view
  const currentWinRate = showTarget 
    ? currentData?.target_demo?.win_rate 
    : currentData?.general_pop?.win_rate
  
  const currentVoteCounts = showTarget
    ? currentData?.target_demo
    : currentData?.general_pop

  return (
    <div className="grpo-dashboard">
      {/* Step Slider */}
      <div className="step-slider-container">
        <div className="step-slider-header">
          <span className="step-label">Training Step</span>
          <span className="step-value">{steps[currentStepIndex]}</span>
        </div>
        <input 
          type="range"
          min={0}
          max={steps.length - 1}
          value={currentStepIndex}
          onChange={(e) => setSelectedStep(steps[parseInt(e.target.value)])}
          className="step-slider"
        />
        <div className="step-ticks">
          <span>0</span>
          <span>{steps[steps.length - 1]}</span>
        </div>
      </div>
      
      {/* Tweet Display - Like Day 18 content showcase */}
      <section className="tweet-showcase">
        <div className="tweet-label">Model's Best Tweet</div>
        <blockquote className="tweet-quote">
          "{displayTweet}"
        </blockquote>
        <div className="tweet-meta">
          <span className="win-rate-badge" style={{ 
            background: currentWinRate > 0.5 ? '#4CAF50' : currentWinRate < 0.5 ? '#F44336' : '#9E9E9E'
          }}>
            {(currentWinRate * 100).toFixed(1)}% win rate vs GPT-4.1
          </span>
          <span className="vote-counts">
            Model: {currentVoteCounts?.model_votes} • GPT-4.1: {currentVoteCounts?.gpt_votes}
          </span>
        </div>
      </section>
      
      {/* Demographic Toggle */}
      <div className="demo-toggle-container">
        <div className="demo-toggle">
          <button 
            className={`toggle-btn ${showTarget ? 'active' : ''}`}
            onClick={() => setShowTarget(true)}
          >
            🎯 Target Demographic
          </button>
          <button 
            className={`toggle-btn ${!showTarget ? 'active' : ''}`}
            onClick={() => setShowTarget(false)}
          >
            🌍 General Population
          </button>
        </div>
        {showTarget && targetDemoDescription && (
          <div className="demo-description">
            {targetDemoDescription}
          </div>
        )}
        {!showTarget && (
          <div className="demo-description">
            Random sample of 10,000 personas from all demographics
          </div>
        )}
      </div>
      
      {/* Map Section */}
      <section className="map-section">
        <h2>How Each State Voted</h2>
        <p className="map-subtitle">
          {showTarget ? 'Target demographic' : 'General population'} preference by state
        </p>
        <MapVisualization stateData={stateData} />
      </section>
      
      {/* Win Rate Over Time (small chart) */}
      <section className="progress-section">
        <h2>Training Progress</h2>
        <div className="mini-chart">
          <svg viewBox="0 0 600 150" className="progress-svg">
            {/* 50% baseline */}
            <line x1="40" y1="75" x2="580" y2="75" stroke="#228B22" strokeWidth="2" strokeDasharray="5,5" opacity="0.5" />
            
            {/* Target line (red) */}
            <polyline
              fill="none"
              stroke="#FF1744"
              strokeWidth="2.5"
              points={steps.map((step, i) => {
                const x = 40 + (i / Math.max(1, steps.length - 1)) * 540
                const y = 140 - (evalResults[step].target_demo.win_rate * 130)
                return `${x},${y}`
              }).join(' ')}
            />
            
            {/* General line (blue) */}
            <polyline
              fill="none"
              stroke="#2196F3"
              strokeWidth="2.5"
              points={steps.map((step, i) => {
                const x = 40 + (i / Math.max(1, steps.length - 1)) * 540
                const y = 140 - (evalResults[step].general_pop.win_rate * 130)
                return `${x},${y}`
              }).join(' ')}
            />
            
            {/* Current position marker */}
            {(() => {
              const x = 40 + (currentStepIndex / Math.max(1, steps.length - 1)) * 540
              return <line x1={x} y1="10" x2={x} y2="140" stroke="#333" strokeWidth="2" strokeDasharray="3,3" />
            })()}
            
            {/* Y-axis labels */}
            <text x="35" y="15" textAnchor="end" fontSize="11" fill="#666">100%</text>
            <text x="35" y="78" textAnchor="end" fontSize="11" fill="#666">50%</text>
            <text x="35" y="145" textAnchor="end" fontSize="11" fill="#666">0%</text>
          </svg>
          <div className="chart-legend">
            <span><span className="dot red"></span> Target Demo</span>
            <span><span className="dot blue"></span> General Pop</span>
            <span><span className="dot green dashed"></span> 50% (equal to GPT-4.1)</span>
          </div>
        </div>
      </section>
      
      {/* GPT Tweet for comparison */}
      <section className="comparison-section">
        <h2>GPT-4.1's Tweet (for comparison)</h2>
        <blockquote className="gpt-quote">
          "{currentData?.gpt_tweets?.[0] || 'No tweet available'}"
        </blockquote>
      </section>
    </div>
  )
}

export default GRPODashboard
