import './SampleRatings.css'

function SampleRatings({ samples }) {
  if (!samples || samples.length === 0) return null

  const getRatingColor = (rating) => {
    if (rating >= 4) return '#4a90e2'
    if (rating >= 3) return '#9ecae1'
    if (rating >= 2) return '#feb24c'
    return '#e24a4a'
  }

  const getRatingLabel = (rating) => {
    if (rating >= 4.5) return 'Strongly Like'
    if (rating >= 3.5) return 'Like'
    if (rating >= 2.5) return 'Neutral'
    if (rating >= 1.5) return 'Dislike'
    return 'Strongly Dislike'
  }

  return (
    <div className="sample-ratings">
      <div className="samples-grid">
        {samples.slice(0, 20).map((sample, idx) => (
          <div key={idx} className="sample-card">
            <div className="sample-header">
              <div className="sample-demographics">
                <span className="demo-item">{sample.sex || 'Unknown'}</span>
                <span className="demo-item">{sample.age} years</span>
                {sample.state && <span className="demo-item">{sample.state}</span>}
              </div>
              <div className="sample-ratings-badge">
                <div 
                  className="rating-badge likeability"
                  style={{ backgroundColor: getRatingColor(sample.likeability) }}
                >
                  {sample.likeability} Like
                </div>
                <div 
                  className="rating-badge emotional"
                  style={{ backgroundColor: getRatingColor(sample.emotional_activation) }}
                >
                  {sample.emotional_activation} Emo
                </div>
              </div>
            </div>
            <div className="sample-reasoning">
              "{sample.reasoning}"
            </div>
            {sample.occupation && (
              <div className="sample-occupation">{sample.occupation}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

export default SampleRatings
