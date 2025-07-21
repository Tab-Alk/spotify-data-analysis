
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Constants
AUDIO_FEATURES = [
    'danceability_pct', 'energy_pct', 'acousticness_pct',
    'speechiness_pct', 'instrumentalness_pct'
]
FEATURE_NAMES = {
    'danceability_pct': 'Danceability',
    'energy_pct': 'Energy',
    'acousticness_pct': 'Acousticness',
    'speechiness_pct': 'Speechiness',
    'instrumentalness_pct': 'Instrumentalness'
}

def load_data():
    """Load and prepare the merged Spotify dataset."""
    try:
        df = pd.read_csv('spotify_merged_2022_2023.csv')
        print(f"Loaded {len(df)} records")
        return df
    except FileNotFoundError:
        print("Error: Could not find 'spotify_merged_2022_2023.csv'")
        return None

def calculate_kpis(df):
    """Calculate KPIs for both years."""
    df = df.copy()
    
    # Chart Sustainability Index (2022)
    if 'weeks_on_chart' in df.columns:
        df['CSI'] = df['weeks_on_chart'] * (101 - df['peak_rank']) / 100
    
    # Streaming Dominance Index (2023)
    if 'streams' in df.columns:
        playlist_term = (df['in_spotify_playlists'].fillna(0) / 1000) ** 0.5
        df['SDI'] = (df['streams'] / 1e6) * playlist_term
    
    return df

def get_success_cohorts(df, percentile=0.9):
    """Identify top-performing songs."""
    df_2022 = df[df['year'] == 2022].copy()
    df_2023 = df[df['year'] == 2023].copy()
    
    top_csi = df_2022[df_2022['CSI'] >= df_2022['CSI'].quantile(percentile)] if not df_2022.empty else None
    top_sdi = df_2023[df_2023['SDI'] >= df_2023['SDI'].quantile(percentile)] if not df_2023.empty else None
    
    return top_csi, top_sdi

def plot_radar_chart(signature_2022, signature_2023):
    """Create a radar chart comparing audio signatures."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=signature_2022.values,
        theta=[FEATURE_NAMES[f] for f in signature_2022.index],
        fill='toself',
        name='2022 (CSI)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=signature_2023.values,
        theta=[FEATURE_NAMES[f] for f in signature_2023.index],
        fill='toself',
        name='2023 (SDI)'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Audio Signatures of Top-Performing Songs"
    )
    
    return fig

def create_success_distribution_plot(df_2022, df_2023):
    """Create side-by-side distribution plots for success metrics."""
    fig = make_subplots(
        rows=1, 
        cols=2, 
        subplot_titles=(
            '2022: Chart Sustainability Index (CSI)', 
            '2023: Streaming Dominance Index (SDI)'
        )
    )
    
    # Add CSI distribution for 2022
    fig.add_trace(
        go.Histogram(
            x=df_2022['CSI'], 
            name='CSI',
            marker_color='#1f77b4',  # Blue
            opacity=0.7
        ), 
        row=1, 
        col=1
    )
    
    # Add SDI distribution for 2023
    fig.add_trace(
        go.Histogram(
            x=df_2023['SDI'], 
            name='SDI',
            marker_color='#ff7f0e',  # Orange
            opacity=0.7
        ), 
        row=1, 
        col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Distribution of Success Metrics by Year",
        xaxis_title="Metric Value",
        yaxis_title="Number of Songs",
        showlegend=False,
        height=500,
        width=1000
    )
    
    # Update x-axis titles for each subplot
    fig.update_xaxes(title_text="CSI Score", row=1, col=1)
    fig.update_xaxes(title_text="SDI Score", row=1, col=2)
    
    return fig

def main():
    """Run the complete analysis."""
    # Load and prepare data
    df = load_data()
    if df is None:
        return
        
    df = calculate_kpis(df)
    
    # Split data by year
    df_2022 = df[df['year'] == 2022].copy()
    df_2023 = df[df['year'] == 2023].copy()
    
    # Get top performers
    top_csi, top_sdi = get_success_cohorts(df)
    
    if top_csi is not None and top_sdi is not None:
        # Calculate audio signatures
        signature_2022 = top_csi[AUDIO_FEATURES].mean()
        signature_2023 = top_sdi[AUDIO_FEATURES].mean()
        
        # Create visualizations
        print("\nCreating visualizations...")
        radar_fig = plot_radar_chart(signature_2022, signature_2023)
        dist_fig = create_success_distribution_plot(df_2022, df_2023)
        
        # Save visualizations
        radar_fig.write_html("audio_signatures_radar.html")
        dist_fig.write_html("success_metrics_distribution.html")
        
        print("\nVisualizations saved as:")
        print("1. audio_signatures_radar.html - Radar chart comparing audio features")
        print("2. success_metrics_distribution.html - Distribution of success metrics")
        
        # Print analysis
        print("\n=== AUDIO FEATURE EVOLUTION ===")
        evolution = signature_2023 - signature_2022
        for feature, change in evolution.items():
            print(f"{FEATURE_NAMES[feature]}: {change:+.1f}%")

if __name__ == "__main__":
    main()
