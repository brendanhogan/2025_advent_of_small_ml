# Persona Simulation Visualizer

NYT-style React frontend for visualizing persona simulation results.

## Setup

```bash
cd frontend
npm install
```

## Development

```bash
npm run dev
```

Opens at http://localhost:3000

## Data Setup

1. Run your simulation: `uv run python batch_simulate.py ...`
2. Aggregate results: `uv run python aggregate_results.py --input run_001 --output viz_data`
3. Copy `viz_data/` to `frontend/public/` (or serve it via a static server)

The app expects these files in `/viz_data/`:
- `overall.json`
- `by_zipcode.json`
- `by_state.json`
- `by_demographics.json`
- `sample_ratings.json`

## Production Build

```bash
npm run build
```

Outputs to `dist/` - serve with any static file server.
