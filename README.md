# zigzag_on_dynamic_social_networks

This repository contains code and data for reproducing analyses in the paper **"The Evolution and Impact of Group Collaboration in Crisis Response"**.

## Project Overview

This project studies the dynamics of volunteer collaboration in crisis response, with a focus on higher-order collaboration structures and their relationship to system effectiveness.

![Figure: Collaboration patterns in volunteer activities](figures/collaboration_patterns.svg)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/anpingzzzz/zigzag_on_dynamic_social_networks.git
cd zigzag_on_dynamic_social_networks
```

### 2. Create and activate environment
```bash
conda create --name zigzag python=3.8.18
conda activate zigzag
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> `dionysus` may require additional local build/runtime setup depending on your platform.

## Usage

### 1. City-level zigzag persistence on volunteer data

```bash
python zigzag.py --T 20
```

Precomputed outputs are under `results/volunteer_data_results/city_level/`.

Common arguments:
- `--input_file`: input parquet file (default `./data/volunteer_acticities.parquet`)
- `--SWL`: sliding window length
- `--OL`: overlap between adjacent windows
- `--T`: minimum group size filter
- `--overlapping_threshold`: overlap threshold for higher-order simplices
- `--end_date`: analysis end date

### 2. Agent-based simulation + zigzag on simulated interactions

```bash
python simulation.py --consider_POI False
```

Main outputs in current working directory:
- `system_effectiveness.pkl`
- `dgms.pkl`
- `agent_info.pkl`

Common arguments:
- `--iters`, `--total_tasks_num`, `--total_agents_num`, `--organizer_percentage`
- `--consider_POI`
- preference weights: `--v1 --v2 --v3 --v4 --v5`
- zigzag params: `--SWL --OL`

### 3. Aggregate replicated simulation results

```bash
python utils/compute_average_from_simulation.py
```

Writes:
- `results/simulation_results/avg_betti_curve_replicate.pkl`
- `results/simulation_results/avg_effectiveness_replicate.pkl`

### 4. MapSwipe visualization and topology analysis

```bash
python utils/mapswipe_visual.py --mode both
python utils/topo_analysis_mapswipe.py
```

Example outputs:
- `figures/mapswipe_global.png`
- `figures/mapswipe_zoom.png`
- `figures/betti_effectiveness_correlation_mapswipe.png`

## File Structure

```text
zigzag_on_dynamic_social_networks/
├── README.md
├── requirements.txt
├── simulation.py
├── zigzag.py
├── empirial_study_analysis.ipynb
├── simulation_analysis.ipynb
├── data/
│   ├── completed_projects_with_coords.csv
│   ├── district_polygon.csv
│   ├── mapswipe_user_contributions_by_date_201909_202510.csv
│   ├── shenzhen_polygon.json
│   ├── street_poi.csv
│   ├── street_polygon.csv
│   ├── street_task_distribution_from_poi.csv
│   ├── street_tast_type_all_same.csv
│   ├── user_experience.json
│   └── volunteer_acticities.parquet
├── figures/
│   ├── collaboration_patterns.svg
│   ├── mapswipe_global.png
│   ├── mapswipe_zoom.png
│   ├── mapswipe_from_empirical_global.png
│   ├── mapswipe_from_empirical_zoom.png
│   └── betti_effectiveness_correlation_mapswipe.png
├── results/
│   ├── mapswipe_data_results/
│   │   └── betti_curve_mapswipe.pkl
│   ├── simulation_results/
│   └── volunteer_data_results/
│       ├── city_level/
│       └── subdistrict_level/
└── utils/
    ├── GetNewMethods.py
    ├── compute_average_from_simulation.py
    ├── mapswipe_visual.py
    ├── topo_analysis_mapswipe.py
    └── util.py
```

> Note: `utils/GetNewMethods.py` includes topological vectorization methods adapted from GUDHI-related implementations and used by this project.
