import { useState } from 'react'
import { ComposableMap, Geographies, Geography, ZoomableGroup } from 'react-simple-maps'
import { scaleLinear } from 'd3-scale'
import './MapVisualization.css'

// Simple US states geo data (you'd use a real GeoJSON in production)
const geoUrl = "https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json"

// State code to name mapping
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

  // Prepare state data - map codes to names
  const states = Object.entries(stateData).map(([stateCode, stats]) => ({
    stateCode,
    stateName: STATE_CODE_TO_NAME[stateCode] || stateCode,
    likeability: stats.likeability_mean,
    emotional: stats.emotional_mean,
    n: stats.n,
  }))

  // Diverging color scale: Red (dislike) -> White (neutral) -> Blue (like)
  // Fixed domain: 3 = strong red, 5.5 = neutral, 8 = strong blue
  const colorScale = scaleLinear()
    .domain([3, 5, 5.5, 6, 8])
    .range(['#c41d24', '#e8a0a0', '#f5f5f5', '#a0c4e8', '#1d4e89'])
    .clamp(true)

  const getStateColor = (stateName) => {
    const state = states.find(s => s.stateName === stateName)
    if (!state) return "#f0f0f0"
    return colorScale(state[selectedMetric])
  }

  const getStateTooltip = (stateName) => {
    const state = states.find(s => s.stateName === stateName)
    if (!state) return null
    return {
      state: state.stateCode,
      stateName: state.stateName,
      likeability: state.likeability.toFixed(2),
      emotional: state.emotional.toFixed(2),
      n: state.n,
    }
  }

  return (
    <div className="map-container">
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
        <div className="map-legend">
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
        </div>
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
                      strokeWidth={0.5}
                      style={{
                        default: { outline: 'none' },
                        hover: { 
                          outline: 'none',
                          fill: '#000',
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
            <div>Likeability: {hoveredState.likeability}</div>
            <div>Emotional: {hoveredState.emotional}</div>
            <div>Sample size: {hoveredState.n.toLocaleString()}</div>
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
            {states.reduce((sum, s) => sum + s.n, 0).toLocaleString()}
          </div>
          <div className="map-stat-label">Total Responses</div>
        </div>
      </div>
    </div>
  )
}

export default MapVisualization
