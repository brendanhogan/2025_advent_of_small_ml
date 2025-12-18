import { useMemo } from 'react'
import './ViolinPlot.css'

// Simple violin plot using SVG
function ViolinPlot({ data, title, color = '#4a90e2', height = 200 }) {
  const stats = useMemo(() => {
    if (!data) return null
    
    // Try to get values array, or reconstruct from distribution
    let values = []
    if (data.values && data.values.length > 0) {
      values = data.values
    } else if (data.dist) {
      // Reconstruct from distribution counts
      values = []
      for (let score = 1; score <= 5; score++) {
        const count = data.dist[score] || 0
        for (let i = 0; i < count; i++) {
          values.push(score)
        }
      }
    }
    
    if (values.length === 0) return null
    
    const mean = data.mean || (values.reduce((a, b) => a + b, 0) / values.length)
    const std = data.std || 0
    
    // Create histogram bins
    const bins = 20
    const min = Math.min(...values)
    const max = Math.max(...values)
    const binWidth = (max - min) / bins
    
    const histogram = Array(bins).fill(0)
    values.forEach(v => {
      const bin = Math.min(Math.floor((v - min) / binWidth), bins - 1)
      histogram[bin]++
    })
    
    const maxCount = Math.max(...histogram)
    
    return { histogram, min, max, mean, std, binWidth, maxCount }
  }, [data])

  if (!stats) return <div className="violin-placeholder">No data</div>

  const width = 300
  const padding = 40
  const plotWidth = width - padding * 2
  const plotHeight = height - padding * 2

  // Normalize histogram to plot width
  const points = stats.histogram.map((count, i) => {
    const x = (count / stats.maxCount) * (plotWidth / 2)
    const y = (i / stats.histogram.length) * plotHeight
    return { x, y, count }
  })

  // Create path for left side
  const leftPath = points.map((p, i) => 
    `${i === 0 ? 'M' : 'L'} ${padding - p.x} ${padding + p.y}`
  ).join(' ')

  // Create path for right side
  const rightPath = points.map((p, i) => 
    `${i === 0 ? 'M' : 'L'} ${padding + p.x} ${padding + p.y}`
  ).join(' ')

  // Close the shape
  const closePath = `L ${padding} ${padding + plotHeight} Z`

  return (
    <div className="violin-plot">
      <div className="violin-title">{title}</div>
      <svg width={width} height={height} className="violin-svg">
        {/* Background */}
        <rect width={width} height={height} fill="#fafafa" />
        
        {/* Violin shape */}
        <path 
          d={`${leftPath} ${closePath}`} 
          fill={color} 
          opacity={0.6}
          stroke={color}
          strokeWidth={1}
        />
        <path 
          d={`${rightPath} ${closePath}`} 
          fill={color} 
          opacity={0.6}
          stroke={color}
          strokeWidth={1}
        />
        
        {/* Mean line */}
        <line
          x1={padding}
          x2={width - padding}
          y1={padding + ((stats.mean - stats.min) / (stats.max - stats.min)) * plotHeight}
          y2={padding + ((stats.mean - stats.min) / (stats.max - stats.min)) * plotHeight}
          stroke="#000"
          strokeWidth={2}
          strokeDasharray="4,4"
        />
        
        {/* Y-axis labels */}
        <text x={padding - 5} y={padding} textAnchor="end" fontSize="10" fill="#666">
          {stats.max.toFixed(1)}
        </text>
        <text x={padding - 5} y={height - padding} textAnchor="end" fontSize="10" fill="#666">
          {stats.min.toFixed(1)}
        </text>
        <text x={padding - 5} y={padding + plotHeight / 2} textAnchor="end" fontSize="10" fill="#666">
          {stats.mean.toFixed(1)}
        </text>
        
        {/* X-axis */}
        <line
          x1={padding}
          x2={width - padding}
          y1={height - padding}
          y2={height - padding}
          stroke="#ccc"
          strokeWidth={1}
        />
      </svg>
      <div className="violin-stats">
        <span>Mean: {stats.mean.toFixed(2)}</span>
        <span>Std: {stats.std.toFixed(2)}</span>
        <span>Range: {stats.min} - {stats.max}</span>
      </div>
    </div>
  )
}

export default ViolinPlot
