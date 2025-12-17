import './StatsDashboard.css'

function StatsDashboard({ overall }) {
  if (!overall || !overall.likeability) return null

  const { likeability, emotional_activation, n } = overall

  // Get max for scaling bars
  const likeMaxPct = Math.max(...Array.from({length: 10}, (_, i) => (likeability.distribution[i + 1] || 0) / n * 100))
  const emoMaxPct = Math.max(...Array.from({length: 10}, (_, i) => (emotional_activation.distribution[i + 1] || 0) / n * 100))

  return (
    <section className="stats-dashboard">
      <div className="stat-card main-stat">
        <div className="stat-value">{likeability.mean.toFixed(2)}</div>
        <div className="stat-label">Average Likeability</div>
        <div className="stat-subtitle">out of 10</div>
        <div className="stat-distribution stat-distribution-10">
          {Array.from({length: 10}, (_, i) => i + 1).map(score => {
            const pct = (likeability.distribution[score] || 0) / n * 100
            const barWidth = likeMaxPct > 0 ? (pct / likeMaxPct) * 100 : 0
            return (
              <div key={score} className="dist-bar">
                <div className="dist-label">{score}</div>
                <div className="dist-bar-container">
                  <div 
                    className="dist-bar-fill"
                    style={{ width: `${barWidth}%` }}
                  />
                </div>
                <div className="dist-count">{pct.toFixed(0)}%</div>
              </div>
            )
          })}
        </div>
      </div>

      <div className="stat-card main-stat">
        <div className="stat-value">{emotional_activation.mean.toFixed(2)}</div>
        <div className="stat-label">Emotional Activation</div>
        <div className="stat-subtitle">out of 10</div>
        <div className="stat-distribution stat-distribution-10">
          {Array.from({length: 10}, (_, i) => i + 1).map(score => {
            const pct = (emotional_activation.distribution[score] || 0) / n * 100
            const barWidth = emoMaxPct > 0 ? (pct / emoMaxPct) * 100 : 0
            return (
              <div key={score} className="dist-bar">
                <div className="dist-label">{score}</div>
                <div className="dist-bar-container">
                  <div 
                    className="dist-bar-fill emotional"
                    style={{ width: `${barWidth}%` }}
                  />
                </div>
                <div className="dist-count">{pct.toFixed(0)}%</div>
              </div>
            )
          })}
        </div>
      </div>

      <div className="stat-card meta-stat">
        <div className="stat-value-large">{n.toLocaleString()}</div>
        <div className="stat-label">Total Personas</div>
        <div className="stat-meta">
          <div>Likeability: {likeability.mean.toFixed(2)} ± {likeability.std.toFixed(2)}</div>
          <div>Emotional: {emotional_activation.mean.toFixed(2)} ± {emotional_activation.std.toFixed(2)}</div>
        </div>
      </div>
    </section>
  )
}

export default StatsDashboard
