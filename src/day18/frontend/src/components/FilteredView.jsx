import { useState, useMemo } from 'react'
import './FilteredView.css'

function FilteredView({ rawRatings }) {
  const [filters, setFilters] = useState({
    sex: 'all',
    ageMin: '',
    ageMax: '',
    state: 'all',
    education: 'all',
    maritalStatus: 'all',
  })

  // Get unique values for dropdowns
  const options = useMemo(() => {
    if (!rawRatings || rawRatings.length === 0) return {}
    
    const unique = (key) => [...new Set(rawRatings.map(r => r[key]).filter(Boolean))].sort()
    
    return {
      sex: unique('sex'),
      state: unique('state'),
      education: unique('education_level'),
      maritalStatus: unique('marital_status'),
    }
  }, [rawRatings])

  // Filter ratings
  const filteredRatings = useMemo(() => {
    if (!rawRatings) return []
    
    return rawRatings.filter(r => {
      if (filters.sex !== 'all' && r.sex !== filters.sex) return false
      if (filters.ageMin && r.age < parseInt(filters.ageMin)) return false
      if (filters.ageMax && r.age > parseInt(filters.ageMax)) return false
      if (filters.state !== 'all' && r.state !== filters.state) return false
      if (filters.education !== 'all' && r.education_level !== filters.education) return false
      if (filters.maritalStatus !== 'all' && r.marital_status !== filters.maritalStatus) return false
      return true
    })
  }, [rawRatings, filters])

  // Compute stats for filtered data
  const stats = useMemo(() => {
    if (filteredRatings.length === 0) return null
    
    const likeability = filteredRatings.map(r => r.likeability)
    const emotional = filteredRatings.map(r => r.emotional_activation)
    
    const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length
    const std = arr => {
      const m = mean(arr)
      return Math.sqrt(arr.reduce((sum, x) => sum + (x - m) ** 2, 0) / arr.length)
    }
    
    return {
      n: filteredRatings.length,
      likeability: {
        mean: mean(likeability),
        std: std(likeability),
        dist: Array.from({length: 10}, (_, i) => likeability.filter(x => x === i + 1).length),
      },
      emotional: {
        mean: mean(emotional),
        std: std(emotional),
        dist: Array.from({length: 10}, (_, i) => emotional.filter(x => x === i + 1).length),
      },
    }
  }, [filteredRatings])

  if (!rawRatings || rawRatings.length === 0) {
    return (
      <div className="filtered-view-empty">
        No raw ratings available.<br/>
        Re-run <code>aggregate_results.py --include-raw</code> to enable filtering.
      </div>
    )
  }

  const formatEducation = (val) => val?.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) || val
  const formatMarital = (val) => val?.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) || val

  return (
    <div className="filtered-view">
      <div className="filters">
        <div className="filter-group">
          <label>Sex</label>
          <select value={filters.sex} onChange={e => setFilters({...filters, sex: e.target.value})}>
            <option value="all">All</option>
            {options.sex?.map(v => <option key={v} value={v}>{v}</option>)}
          </select>
        </div>
        
        <div className="filter-group age-range">
          <label>Age Range</label>
          <div className="age-inputs">
            <input 
              type="number" 
              placeholder="Min" 
              value={filters.ageMin}
              onChange={e => setFilters({...filters, ageMin: e.target.value})}
            />
            <span className="age-separator">to</span>
            <input 
              type="number" 
              placeholder="Max"
              value={filters.ageMax}
              onChange={e => setFilters({...filters, ageMax: e.target.value})}
            />
          </div>
        </div>
        
        <div className="filter-group">
          <label>State</label>
          <select value={filters.state} onChange={e => setFilters({...filters, state: e.target.value})}>
            <option value="all">All States</option>
            {options.state?.map(v => <option key={v} value={v}>{v}</option>)}
          </select>
        </div>
        
        <div className="filter-group">
          <label>Education</label>
          <select value={filters.education} onChange={e => setFilters({...filters, education: e.target.value})}>
            <option value="all">All Education Levels</option>
            {options.education?.map(v => <option key={v} value={v}>{formatEducation(v)}</option>)}
          </select>
        </div>
        
        <div className="filter-group">
          <label>Marital Status</label>
          <select value={filters.maritalStatus} onChange={e => setFilters({...filters, maritalStatus: e.target.value})}>
            <option value="all">All</option>
            {options.maritalStatus?.map(v => <option key={v} value={v}>{formatMarital(v)}</option>)}
          </select>
        </div>
        
        <button 
          className="reset-btn" 
          onClick={() => setFilters({sex: 'all', ageMin: '', ageMax: '', state: 'all', education: 'all', maritalStatus: 'all'})}
        >
          Reset All
        </button>
      </div>

      {stats && (
        <div className="filtered-results">
          <div className="filtered-summary">
            <span className="count">{stats.n.toLocaleString()}</span>
            personas selected ({((stats.n / rawRatings.length) * 100).toFixed(1)}% of {rawRatings.length.toLocaleString()} total)
          </div>
          
          <div className="filtered-histograms">
            <div className="histogram-panel">
              <h4>Likeability Distribution (1-10)</h4>
              <div className="histogram-display">
                {Array.from({length: 10}, (_, i) => i + 1).map((score, i) => {
                  const pct = (stats.likeability.dist[i] / stats.n) * 100
                  const maxPct = Math.max(...stats.likeability.dist) / stats.n * 100
                  const barHeight = maxPct > 0 ? (pct / maxPct) * 130 : 0
                  // Color gradient: red (1-3) -> yellow (4-6) -> green (7-10)
                  const getColor = (s) => {
                    if (s <= 3) return '#C62828'
                    if (s <= 6) return '#F9A825'
                    return '#2E7D32'
                  }
                  return (
                    <div key={score} className="hist-bar-group">
                      <div className="hist-bar-wrapper">
                        <div 
                          className="hist-bar"
                          style={{
                            height: `${Math.max(barHeight, 4)}px`,
                            backgroundColor: getColor(score)
                          }}
                          title={`${stats.likeability.dist[i].toLocaleString()} responses`}
                        />
                      </div>
                      <div className="hist-label">{score}</div>
                      <div className="hist-pct">{pct.toFixed(0)}%</div>
                    </div>
                  )
                })}
              </div>
              <div className="histogram-stats">
                Mean: {stats.likeability.mean.toFixed(2)} &nbsp;|&nbsp; Std Dev: {stats.likeability.std.toFixed(2)}
              </div>
            </div>
            
            <div className="histogram-panel">
              <h4>Emotional Activation (1-10)</h4>
              <div className="histogram-display">
                {Array.from({length: 10}, (_, i) => i + 1).map((score, i) => {
                  const pct = (stats.emotional.dist[i] / stats.n) * 100
                  const maxPct = Math.max(...stats.emotional.dist) / stats.n * 100
                  const barHeight = maxPct > 0 ? (pct / maxPct) * 130 : 0
                  // Color gradient: blue (low) -> orange (mid) -> red (high)
                  const getColor = (s) => {
                    if (s <= 3) return '#5C6BC0'
                    if (s <= 6) return '#FFB300'
                    return '#D84315'
                  }
                  return (
                    <div key={score} className="hist-bar-group">
                      <div className="hist-bar-wrapper">
                        <div 
                          className="hist-bar"
                          style={{
                            height: `${Math.max(barHeight, 4)}px`,
                            backgroundColor: getColor(score)
                          }}
                          title={`${stats.emotional.dist[i].toLocaleString()} responses`}
                        />
                      </div>
                      <div className="hist-label">{score}</div>
                      <div className="hist-pct">{pct.toFixed(0)}%</div>
                    </div>
                  )
                })}
              </div>
              <div className="histogram-stats">
                Mean: {stats.emotional.mean.toFixed(2)} &nbsp;|&nbsp; Std Dev: {stats.emotional.std.toFixed(2)}
              </div>
            </div>
          </div>
          
          <div className="filtered-samples">
            <h4>Sample Responses from Selection</h4>
            <div className="samples-list">
              {filteredRatings.slice(0, 8).map((r, i) => (
                <div key={i} className="sample-item">
                  <div className="sample-meta">
                    <span>{r.sex}, {r.age} years old</span>
                    <span>{r.city}, {r.state}</span>
                    <span className="rating-badge">
                      <span className="badge" style={{background: r.likeability >= 4 ? '#C8E6C9' : r.likeability <= 2 ? '#FFCDD2' : '#E0E0E0'}}>
                        Like: {r.likeability}
                      </span>
                      <span className="badge" style={{background: r.emotional_activation >= 4 ? '#FFE0B2' : '#E0E0E0'}}>
                        Emo: {r.emotional_activation}
                      </span>
                    </span>
                  </div>
                  <div className="sample-reasoning">"{r.reasoning}"</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default FilteredView
