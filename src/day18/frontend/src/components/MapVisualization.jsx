import { useState } from 'react'
import { ComposableMap, Geographies, Geography, ZoomableGroup } from 'react-simple-maps'
import { scaleLinear } from 'd3-scale'
import './MapVisualization.css'

const geoUrl = "https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json"

const STATE_CODE_TO_NAME = {
  'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
  'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
  'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
  'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
  'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
  'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
  'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
  'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
  'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
  'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
}

function MapVisualization({ zipcodeData, stateData }) {
  const [selectedMetric, setSelectedMetric] = useState('likeability')
  const [hoveredState, setHoveredState] = useState(null)

  if (!stateData || Object.keys(stateData).length === 0) {
    return <div className="map-placeholder">No geographic data available</div>
  }

  // Detect if this is GRPO win rate data (has win_rate field)
  const firstState = Object.values(stateData)[0]
  const isGRPOMode = firstState && 'win_rate' in firstState

  // Prepare state data - map codes to names
  const states = Object.entries(stateData).map(([stateCode, stats]) => ({
    stateCode,
    stateName: STATE_CODE_TO_NAME[stateCode] || stateCode,
    likeability: stats.likeability_mean,
    emotional: stats.emotional_mean,
    win_rate: stats.win_rate,
    model_votes: stats.model_votes,
    gpt_votes: stats.gpt_votes,
    n: stats.n,
  }))

  // NYT-style political colors for GRPO mode
  // Red (GPT wins) -> Grey (50/50) -> Blue (Model wins)
  const grpoColorScale = scaleLinear()
    .domain([0, 0.35, 0.5, 0.65, 1])
    .range(['#b2182b', '#ef8a62', '#e0e0e0', '#67a9cf', '#2166ac'])
    .clamp(true)

  // Original Day 18 color scale
  const ratingColorScale = scaleLinear()
    .domain([3, 5, 5.5, 6, 8])
    .range(['#c41d24', '#e8a0a0', '#f5f5f5', '#a0c4e8', '#1d4e89'])
    .clamp(true)

  const getStateColor = (stateName) => {
    const state = states.find(s => s.stateName === stateName)
    if (!state) return "#f0f0f0"
    
    if (isGRPOMode) {
      return grpoColorScale(state.win_rate)
    }
    return ratingColorScale(state[selectedMetric])
  }

  const getStateTooltip = (stateName) => {
    const state = states.find(s => s.stateName === stateName)
    if (!state) return null
    
    if (isGRPOMode) {
      return {
        state: state.stateCode,
        stateName: state.stateName,
        win_rate: state.win_rate,
        model_votes: state.model_votes,
        gpt_votes: state.gpt_votes,
        n: state.n,
        isGRPO: true,
      }
    }
    
    return {
      state: state.stateCode,
      stateName: state.stateName,
      likeability: state.likeability?.toFixed(2),
      emotional: state.emotional?.toFixed(2),
      n: state.n,
      isGRPO: false,
    }
  }

  return (
    <div className="map-container">
      {/* Only show metric selector for Day 18 mode */}
      {!isGRPOMode && (
        <div className="map-controls">
          <div className="metric-selector">
            <button
              className={selectedMetric === 'likeability' ? 'active' : ''}
              onClick={() => setSelectedMetric('likeability')}
            >
              Likeability
            </button>
            <button
              className={selectedMetric === 'emotional' ? 'active' : ''}
              onClick={() => setSelectedMetric('emotional')}
            >
              Emotional Activation
            </button>
          </div>
        </div>
      )}
      
      {/* Legend - different for each mode */}
      <div className="map-legend">
        {isGRPOMode ? (
          <>
            <div className="legend-scale">
              <span className="legend-value">0%</span>
              <span className="legend-value">50%</span>
              <span className="legend-value">100%</span>
            </div>
            <div className="legend-gradient-political" />
            <div className="legend-labels">
              <span>GPT-4.1</span>
              <span>Split</span>
              <span>Model</span>
            </div>
          </>
        ) : (
          <>
            <div className="legend-scale">
              <span className="legend-value">3</span>
              <span className="legend-value">5</span>
              <span className="legend-value">8</span>
            </div>
            <div className="legend-gradient-diverging" />
            <div className="legend-labels">
              <span>Dislike</span>
              <span>Neutral</span>
              <span>Like</span>
            </div>
          </>
        )}
      </div>

      <div className="map-wrapper">
        <ComposableMap
          projection="geoAlbersUsa"
          width={1000}
          height={600}
        >
          <ZoomableGroup>
            <Geographies geography={geoUrl}>
              {({ geographies }) =>
                geographies.map((geo) => {
                  const stateName = geo.properties.name
                  const tooltip = getStateTooltip(stateName)
                  const color = getStateColor(stateName)
                  
                  return (
                    <Geography
                      key={geo.rsmKey}
                      geography={geo}
                      fill={color}
                      stroke="#fff"
                      strokeWidth={0.75}
                      style={{
                        default: { outline: 'none' },
                        hover: { 
                          outline: 'none',
                          stroke: '#000',
                          strokeWidth: 2,
                          cursor: 'pointer'
                        },
                        pressed: { outline: 'none' }
                      }}
                      onMouseEnter={() => setHoveredState(tooltip)}
                      onMouseLeave={() => setHoveredState(null)}
                    />
                  )
                })
              }
            </Geographies>
          </ZoomableGroup>
        </ComposableMap>
      </div>

      {hoveredState && (
        <div className="map-tooltip">
          <div className="tooltip-state">{hoveredState.stateName || hoveredState.state}</div>
          <div className="tooltip-stats">
            {hoveredState.isGRPO ? (
              <>
                <div className="tooltip-winrate" style={{
                  color: hoveredState.win_rate > 0.5 ? '#2166ac' : hoveredState.win_rate < 0.5 ? '#b2182b' : '#666'
                }}>
                  {(hoveredState.win_rate * 100).toFixed(1)}% Model
                </div>
                <div>Model: {hoveredState.model_votes} votes</div>
                <div>GPT-4.1: {hoveredState.gpt_votes} votes</div>
              </>
            ) : (
              <>
                <div>Likeability: {hoveredState.likeability}</div>
                <div>Emotional: {hoveredState.emotional}</div>
              </>
            )}
            <div className="tooltip-n">n = {hoveredState.n?.toLocaleString()}</div>
          </div>
        </div>
      )}

      <div className="map-stats">
        <div className="map-stat-item">
          <div className="map-stat-value">{states.length}</div>
          <div className="map-stat-label">States</div>
        </div>
        <div className="map-stat-item">
          <div className="map-stat-value">
            {states.reduce((sum, s) => sum + (s.n || 0), 0).toLocaleString()}
          </div>
          <div className="map-stat-label">{isGRPOMode ? 'Total Votes' : 'Total Responses'}</div>
        </div>
        {isGRPOMode && (
          <>
            <div className="map-stat-item model">
              <div className="map-stat-value">
                {states.reduce((sum, s) => sum + (s.model_votes || 0), 0).toLocaleString()}
              </div>
              <div className="map-stat-label">Model Votes</div>
            </div>
            <div className="map-stat-item gpt">
              <div className="map-stat-value">
                {states.reduce((sum, s) => sum + (s.gpt_votes || 0), 0).toLocaleString()}
              </div>
              <div className="map-stat-label">GPT-4.1 Votes</div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

export default MapVisualization
