import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import './DemographicBreakdown.css'

function DemographicBreakdown({ demographics }) {
  if (!demographics) return null

  const renderChart = (title, data, key) => {
    if (!data || Object.keys(data).length === 0) return null

    const chartData = Object.entries(data)
      .map(([name, stats]) => ({
        name: formatLabel(name),
        likeability: stats.likeability_mean,
        emotional: stats.emotional_mean,
        likeability_std: stats.likeability_std || 0,
        emotional_std: stats.emotional_std || 0,
        likeability_dist: stats.likeability_dist || {},
        emotional_dist: stats.emotional_dist || {},
        n: stats.n,
      }))
      .sort((a, b) => b.likeability - a.likeability)

    const colors = ['#4a90e2', '#357abd', '#2171b5', '#08519c']

    return (
      <div className="demographic-chart">
        <h3>{title}</h3>
        
        {/* Histograms showing distribution (1-10 scale) */}
        <div className="histogram-section">
          <h4>Likeability Distribution (1-10)</h4>
          <div className="histograms-grid">
            {chartData.slice(0, 6).map((item, idx) => {
              const maxCount = Math.max(...Array.from({length: 10}, (_, i) => item.likeability_dist[i + 1] || 0))
              return (
                <div key={idx} className="histogram-item">
                  <div className="histogram-title">{item.name}</div>
                  <div className="histogram-bars histogram-bars-10">
                    {Array.from({length: 10}, (_, i) => i + 1).map(score => {
                      const count = item.likeability_dist[score] || 0
                      const barPct = maxCount > 0 ? (count / maxCount) * 100 : 0
                      const getColor = (s) => s <= 3 ? '#C62828' : s <= 6 ? '#F9A825' : '#2E7D32'
                      return (
                        <div key={score} className="histogram-bar-group">
                          <div className="histogram-bar-container">
                            <div 
                              className="histogram-bar"
                              style={{ 
                                height: `${Math.max(barPct, 3)}%`,
                                backgroundColor: getColor(score)
                              }}
                              title={`${count} responses`}
                            />
                          </div>
                          <div className="histogram-bar-label">{score}</div>
                        </div>
                      )
                    })}
                  </div>
                  <div className="histogram-meta">
                    μ={item.likeability.toFixed(1)} σ={item.likeability_std.toFixed(1)} (n={item.n.toLocaleString()})
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        <div className="histogram-section">
          <h4>Emotional Activation (1-10)</h4>
          <div className="histograms-grid">
            {chartData.slice(0, 6).map((item, idx) => {
              const maxCount = Math.max(...Array.from({length: 10}, (_, i) => item.emotional_dist[i + 1] || 0))
              return (
                <div key={idx} className="histogram-item">
                  <div className="histogram-title">{item.name}</div>
                  <div className="histogram-bars histogram-bars-10">
                    {Array.from({length: 10}, (_, i) => i + 1).map(score => {
                      const count = item.emotional_dist[score] || 0
                      const barPct = maxCount > 0 ? (count / maxCount) * 100 : 0
                      const getColor = (s) => s <= 3 ? '#5C6BC0' : s <= 6 ? '#FFB300' : '#D84315'
                      return (
                        <div key={score} className="histogram-bar-group">
                          <div className="histogram-bar-container">
                            <div 
                              className="histogram-bar"
                              style={{ 
                                height: `${Math.max(barPct, 3)}%`,
                                backgroundColor: getColor(score)
                              }}
                              title={`${count} responses`}
                            />
                          </div>
                          <div className="histogram-bar-label">{score}</div>
                        </div>
                      )
                    })}
                  </div>
                  <div className="histogram-meta">
                    μ={item.emotional.toFixed(1)} σ={item.emotional_std.toFixed(1)} (n={item.n.toLocaleString()})
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Bar chart for means with error bars */}
        <div className="bar-chart-container">
          <h4>Mean Ratings (with std dev)</h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
              <XAxis 
                dataKey="name" 
                angle={-45} 
                textAnchor="end" 
                height={100}
                tick={{ fontSize: 12 }}
              />
              <YAxis domain={[1, 10]} tick={{ fontSize: 12 }} />
              <Tooltip 
                formatter={(value, name, props) => {
                  if (name === 'likeability') {
                    return [`${value.toFixed(2)} ± ${props.payload.likeability_std.toFixed(2)}`, 'Likeability']
                  }
                  return [`${value.toFixed(2)} ± ${props.payload.emotional_std.toFixed(2)}`, 'Emotional']
                }}
                contentStyle={{ backgroundColor: '#fff', border: '1px solid #ccc' }}
              />
              <Bar dataKey="likeability" fill="#4a90e2" name="Likeability">
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill="#4a90e2" />
                ))}
              </Bar>
              <Bar dataKey="emotional" fill="#e24a4a" name="Emotional" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        
        <div className="chart-meta">
          {chartData.map((item, idx) => (
            <div key={idx} className="chart-meta-item">
              <span className="meta-name">{item.name}:</span>
              <span className="meta-value">
                {item.n.toLocaleString()} responses | 
                Like: {item.likeability.toFixed(2)}±{item.likeability_std.toFixed(2)} | 
                Emo: {item.emotional.toFixed(2)}±{item.emotional_std.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      </div>
    )
  }

  const formatLabel = (label) => {
    return label
      .replace(/_/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase())
  }

  return (
    <div className="demographic-breakdown">
      {demographics.by_sex && renderChart('By Sex', demographics.by_sex, 'sex')}
      {demographics.by_age_bracket && renderChart('By Age', demographics.by_age_bracket, 'age')}
      {demographics.by_education && renderChart('By Education', demographics.by_education, 'education')}
      {demographics.by_marital_status && renderChart('By Marital Status', demographics.by_marital_status, 'marital')}
      {demographics.by_occupation && renderChart('By Occupation', demographics.by_occupation, 'occupation')}
    </div>
  )
}

export default DemographicBreakdown
