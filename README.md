# Spotify Data Analysis (2022-2023)

This project analyzes Spotify's top charts data from 2022 and 2023, comparing audio features and success metrics between the two years.

## Features

- Data cleaning and standardization of Spotify datasets
- Calculation of custom success metrics (CSI and SDI)
- Interactive visualizations of audio feature evolution
- Comparative analysis of chart performance vs. streaming dominance

## Requirements

- Python 3.7+
- pandas
- plotly
- matplotlib
- seaborn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/REPOSITORY-NAME.git
   cd spotify-data-analysis
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the analysis:
```bash
python spotify_analysis.py
```

This will generate:
- `audio_signatures_radar.html`: Interactive radar chart of audio features
- `success_metrics_distribution.html`: Distribution of success metrics

## Data Sources
- Spotify Charts data for 2022 and 2023
- Audio features provided by Spotify's Web API

## License
[MIT](LICENSE)
