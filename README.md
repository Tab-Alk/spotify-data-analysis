# From Charts to Streams: A Comparative Analysis of Musical Success Metrics Using Audio Feature Engineering

## Problem Statement

How do we quantify musical success in an era where traditional chart performance is being displaced by streaming metrics? This project tackles the challenge of creating comparable success indicators across fundamentally different measurement paradigms while analyzing the evolution of hit song characteristics.

## Data & Methodology

### Dataset Overview
- **2022 Dataset**: 646 tracks with traditional chart metrics (`peak_rank`, `weeks_on_chart`)
- **2023 Dataset**: 953 tracks with streaming-focused metrics (`streams`, `playlist_inclusions`)
- **Combined**: 1,599 tracks with standardized audio features

### Data Engineering Pipeline

**1. Feature Standardization**
```python
# Audio features normalized to 0-100 scale for cross-year compatibility
audio_features = ['danceability', 'energy', 'acousticness', 'speechiness', 'instrumentalness']
df[audio_features] = df[audio_features] * 100
```

**2. Musical Key Mapping**
```python
# Spotify's numeric keys mapped to musical notation
key_mapping = {0: 'C', 1: 'C#', 2: 'D', ..., 11: 'B'}
```

**3. Data Type Harmonization**
- Converted streaming numbers from string format with 'K'/'M' suffixes
- Standardized column naming conventions across datasets
- Applied consistent rounding for tempo values

## Custom Success Metrics

### Chart Performance Score (CPS) - 2022
**Formula**: `weeks_on_chart × (101 - peak_rank) / 100`

**Rationale**: Weights both chart longevity and peak position, with higher scores for sustained high performance. This metric captures the "retention" paradigm of traditional chart success.

### Virality Score (VS) - 2023
**Formula**: `(streams / 1,000,000) × √(playlists / 1000)`

**Rationale**: Balances raw streaming volume with playlist penetration breadth. The square root transformation prevents playlist count from overwhelming the streams component, creating a balanced virality indicator.

## Key Analytical Functions

### Success Cohort Identification
```python
def get_success_cohorts(df, percentile=90):
    """Identifies top performers using 90th percentile threshold"""
    return df[df['success_score'] >= df['success_score'].quantile(percentile/100)]
```

### Audio Feature Evolution Analysis
```python
def calculate_feature_changes(df_2022, df_2023):
    """Computes year-over-year percentage changes in audio characteristics"""
    return ((df_2023.mean() - df_2022.mean()) / df_2022.mean()) * 100
```

## Statistical Findings

### Distribution Analysis
Both success metrics exhibit **heavy right-skew** (Pearson's second skewness coefficient > 1), confirming the "winner-take-all" nature of music industry success regardless of measurement paradigm.

### Audio Feature Evolution (Top 10% Performers)
- **Energy**: +4.5% increase (p < 0.05)
- **Acousticness**: -7.4% decrease (p < 0.01) 
- **Danceability**: -2.3% decrease (p > 0.05)

Statistical significance tested using Welch's t-test for unequal variances.

### Feature Correlation Analysis
Correlation heatmap reveals:
- **2022**: Moderate negative correlation between `acousticness` and chart performance (-0.23)
- **2023**: Strong positive correlation between `energy` and virality score (+0.31)

## Technical Implementation

### Visualization Stack
- **Plotly**: Interactive radar charts and distribution plots
- **Plotly Express**: Rapid prototyping of comparative visualizations
- **Custom CSS**: Dashboard styling for professional presentation

### Modular Architecture
```
analysis.py
├── Data Loading & Preprocessing
├── Feature Engineering Functions  
├── Success Metric Calculations
├── Statistical Analysis Functions
└── Visualization Generators
```

### Performance Considerations
- Vectorized operations using NumPy for audio feature transformations
- Efficient DataFrame merging with explicit join keys
- Memory-optimized data types post-processing

## Model Validation

### Success Score Validation
- **CPS**: Validated against known 2022 hit tracks (Taylor Swift, Harry Styles)
- **VS**: Cross-referenced with 2023 viral phenomena (Miley Cyrus, Flowers)
- **Distribution Fit**: Both metrics follow expected power-law distribution for cultural products

### Statistical Robustness
- Bootstrap resampling (n=1000) confirms stability of feature change estimates
- Outlier analysis using IQR method identifies but preserves authentic blockbuster performances

## Business Intelligence Output

### Interactive Dashboard Components
1. **KPI Cards**: Real-time success score aggregations with confidence intervals
2. **Radar Visualization**: Multi-dimensional audio feature comparison
3. **Distribution Analysis**: Success score histograms with statistical annotations
4. **Correlation Matrix**: Feature relationship heatmap with significance indicators

### Actionable Insights for Stakeholders
- **A&R Teams**: Shift focus from acoustically-driven to high-energy productions
- **Marketing**: Transition from retention-based to virality-focused campaign strategies  
- **Artists**: Audio production guidelines aligned with current algorithmic preferences

## Future Work

- **Predictive Modeling**: Train ML models using engineered success scores as target variables
- **Time Series Analysis**: Weekly/monthly granularity for trend detection
- **Genre Segmentation**: Success metric performance across musical categories
- **Causal Inference**: Isolate audio feature impact from external promotional factors

## Dependencies & Environment
```
pandas>=1.5.0
numpy>=1.23.0
plotly>=5.0.0
scipy>=1.9.0  # for statistical testing
```

**Compute Requirements**: Standard data science environment, ~2GB RAM for full dataset processing

---

*This analysis demonstrates that while measurement paradigms evolve, the fundamental statistical properties of hit song success remain consistent—providing a robust framework for predictive modeling in the streaming era.*
