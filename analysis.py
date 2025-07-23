import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def load_and_merge_spotify_data():
    # Load datasets
    df_2022 = pd.read_csv('spotify_top_charts_22.csv', encoding_errors='replace')
    df_2023 = pd.read_csv('spotify-2023.csv', encoding_errors='replace', encoding='latin-1')
    
    # 2022 Data Standardization
    df_2022_std = df_2022.copy()
    
    # Clean and transform 2022 data
    key_mapping = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
                  6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}
    df_2022_std['key'] = df_2022_std['key'].map(key_mapping)
    df_2022_std['tempo'] = df_2022_std['tempo'].round().astype(int)
    df_2022_std['mode'] = df_2022_std['mode'].map({1: 'Major', 0: 'Minor'})
    df_2022_std['year'] = 2022
    df_2022_std['artist_name'] = df_2022_std['artist_names']
    
    # Convert audio features to percentage
    audio_features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                     'instrumentalness', 'liveness']
    for feature in audio_features:
        df_2022_std[f'{feature}_pct'] = (df_2022_std[feature] * 100).round().astype(int)
    
    # 2023 Data Standardization
    df_2023_std = df_2023.copy()
    df_2023_std['year'] = 2023
    df_2023_std['artist_name'] = df_2023_std['artist(s)_name']
    
    # Rename audio feature columns using a dictionary
    audio_cols = {
        'danceability_%': 'danceability_pct',
        'energy_%': 'energy_pct',
        'speechiness_%': 'speechiness_pct',
        'acousticness_%': 'acousticness_pct',
        'instrumentalness_%': 'instrumentalness_pct',
        'liveness_%': 'liveness_pct',
        'valence_%': 'valence_pct'
    }
    df_2023_std = df_2023_std.rename(columns=audio_cols)
    
    # Standardize tempo column
    df_2023_std['tempo'] = df_2023_std['bpm']
    
    # Clean streams column
    if df_2023_std['streams'].dtype == 'object':
        df_2023_std['streams'] = pd.to_numeric(df_2023_std['streams'].str.replace(',', ''), errors='coerce')
    
    # Define required columns for each year
    required_cols_2022 = ['year', 'track_name', 'artist_name', 
                         'danceability_pct', 'energy_pct', 'speechiness_pct',
                         'acousticness_pct', 'instrumentalness_pct', 'liveness_pct',
                         'key', 'mode', 'tempo', 'peak_rank', 'weeks_on_chart', 
                         'loudness', 'duration_ms']

    required_cols_2023 = ['year', 'track_name', 'artist_name', 
                         'danceability_pct', 'energy_pct', 'speechiness_pct',
                         'acousticness_pct', 'instrumentalness_pct', 'liveness_pct',
                         'valence_pct', 'key', 'mode', 'tempo', 'streams', 
                         'in_spotify_playlists', 'in_spotify_charts', 'released_year']
    
    # Only select columns that exist in each dataframe
    df_2022_final = df_2022_std[[col for col in required_cols_2022 if col in df_2022_std.columns]]
    df_2023_final = df_2023_std[[col for col in required_cols_2023 if col in df_2023_std.columns]]
    
    # Get common columns for the final output
    common_columns = list(set(required_cols_2022 + required_cols_2023))
    
    # Combine datasets
    df_merged = pd.concat([df_2022_final, df_2023_final], ignore_index=True)
    
    # Data quality checks
    print("\n" + "="*50)
    print("DATA QUALITY SUMMARY")
    print("="*50)
    print(f"Total records: {len(df_merged)}")
    print(f"2022 records: {len(df_merged[df_merged['year'] == 2022])}")
    print(f"2023 records: {len(df_merged[df_merged['year'] == 2023])}")
    
    print(f"\nAudio features coverage:")
    audio_cols = ['danceability_pct', 'energy_pct', 'acousticness_pct']
    for col in audio_cols:
        non_null = len(df_merged[df_merged[col].notna()])
        print(f"  {col}: {non_null}/{len(df_merged)} ({non_null/len(df_merged)*100:.1f}%)")
    
    print(f"\nSuccess metrics coverage:")
    print(f"  peak_rank: {len(df_merged[df_merged['peak_rank'].notna()])} records (2022 only)")
    print(f"  streams: {len(df_merged[df_merged['streams'].notna()])} records (2023 only)")
    
    return df_merged

# Step 7: Execute the merge
def standardize_integer_columns(df):

    # Store original dtypes for reference
    original_dtypes = {}
    
    # Columns to convert to Int64 (nullable integer)
    int_columns = [
        'peak_rank', 'weeks_on_chart', 'duration_ms',  # 2022 columns
        'streams', 'in_spotify_playlists', 'in_spotify_charts',  # 2023 columns
        'released_year', 'valence_pct'  # Both years
    ]
    
    for col in int_columns:
        if col in df.columns:
            # Store original dtype
            original_dtypes[col] = str(df[col].dtype)
            
            # Convert to numeric first (handles any string representations)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert to nullable integer type
            df[col] = df[col].astype('Int64')
    
    return df, original_dtypes



###########      ANALYSIS      ###########



# Constants for analysis
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


def calculate_kpis(df):
    df = df.copy()
    
    # Chart Performance Score (2022)
    if 'weeks_on_chart' in df.columns and 'peak_rank' in df.columns:
        df['CPS'] = df['weeks_on_chart'] * (101 - df['peak_rank']) / 100
    
    # Virality Score (2023)
    if 'streams' in df.columns and 'in_spotify_playlists' in df.columns:
        playlist_term = (df['in_spotify_playlists'].fillna(0) / 1000) ** 0.5
        df['VS'] = (df['streams'] / 1e6) * playlist_term
    
    return df


def get_success_cohorts(df, percentile=0.9):
    """Identify top-performing songs."""
    df_2022 = df[df['year'] == 2022].copy()
    df_2023 = df[df['year'] == 2023].copy()
    
    top_cps = df_2022[df_2022['CPS'] >= df_2022['CPS'].quantile(percentile)] if not df_2022.empty and 'CPS' in df_2022.columns else None
    top_vs = df_2023[df_2023['VS'] >= df_2023['VS'].quantile(percentile)] if not df_2023.empty and 'VS' in df_2023.columns else None
    
    return top_cps, top_vs


def plot_radar_chart(signature_2022, signature_2023):
    """Create a radar chart comparing audio signatures."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=signature_2022.values,
        theta=[FEATURE_NAMES[f] for f in signature_2022.index],
        fill='toself',
        name='2022 (CPS)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=signature_2023.values,
        theta=[FEATURE_NAMES[f] for f in signature_2023.index],
        fill='toself',
        name='2023 (VS)'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Audio Signatures of Top-Performing Songs"
    )
    
    return fig


def create_correlation_heatmap(df_2022, df_2023):
    """Create a heatmap showing correlation between audio features and success metrics."""
    # Create a list to store correlation data for each year
    correlations = []
    
    # Process 2022 data
    if not df_2022.empty and 'CPS' in df_2022.columns:
        # Select audio features and CPS
        df_corr_2022 = df_2022[AUDIO_FEATURES + ['CPS']].copy()
        # Calculate correlation with CPS
        corr_2022 = df_corr_2022.corr()['CPS'].drop('CPS')
        corr_2022.name = '2022 (CPS)'
        correlations.append(corr_2022)
    
    # Process 2023 data
    if not df_2023.empty and 'VS' in df_2023.columns:
        # Select audio features and VS
        df_corr_2023 = df_2023[AUDIO_FEATURES + ['VS']].copy()
        # Calculate correlation with VS
        corr_2023 = df_corr_2023.corr()['VS'].drop('VS')
        corr_2023.name = '2023 (VS)'
        correlations.append(corr_2023)
    
    if not correlations:
        print("Warning: Not enough data to create correlation heatmap")
        return None
    
    # Combine correlations into a single DataFrame
    corr_df = pd.concat(correlations, axis=1)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=[FEATURE_NAMES.get(f, f) for f in corr_df.index],
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Correlation'),
        text=corr_df.round(2).values,
        texttemplate="%{text}",
        textfont={"size":10}
    ))
    
    # Update layout
    fig.update_layout(
        title='Audio Feature Correlation with Success Metrics',
        xaxis_title='Success Metric',
        yaxis_title='Audio Feature',
        height=400 + len(corr_df) * 30,
        width=800,
        margin=dict(l=150, r=50, t=100, b=100)
    )
    
    return fig


def create_success_distribution_plot(df_2022, df_2023):
    """Create side-by-side distribution plots for success metrics."""
    fig = make_subplots(
        rows=1, 
        cols=2, 
        subplot_titles=(
            '2022: Chart Performance Score (CPS)', 
            '2023: Virality Score (VS)'
        )
    )
    
    # Add CPS distribution for 2022 if data is available
    if not df_2022.empty and 'CPS' in df_2022.columns:
        fig.add_trace(
            go.Histogram(
                x=df_2022['CPS'], 
                name='CPS',
                marker_color='#1f77b4',  # Blue
                opacity=0.7
            ), 
            row=1, 
            col=1
        )
    
    # Add VS distribution for 2023 if data is available
    if not df_2023.empty and 'VS' in df_2023.columns:
        fig.add_trace(
            go.Histogram(
                x=df_2023['VS'], 
                name='VS',
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
    if not df_2022.empty and 'CPS' in df_2022.columns:
        fig.update_xaxes(title_text="CPS Score", row=1, col=1)
    if not df_2023.empty and 'VS' in df_2023.columns:
        fig.update_xaxes(title_text="VS Score", row=1, col=2)
    
    return fig


def run_analysis(df):
    """Run the complete analysis on merged data."""
    print("\n" + "="*50)
    print("RUNNING ANALYSIS")
    print("="*50)
    
    # Calculate KPIs
    df = calculate_kpis(df)
    
    # Split data by year
    df_2022 = df[df['year'] == 2022].copy()
    df_2023 = df[df['year'] == 2023].copy()
    
    # Get top performers
    top_csi, top_sdi = get_success_cohorts(df)
    
    if top_csi is not None and top_sdi is not None and not top_csi.empty and not top_sdi.empty:
        # Calculate audio signatures
        signature_2022 = top_csi[AUDIO_FEATURES].mean()
        signature_2023 = top_sdi[AUDIO_FEATURES].mean()
        
        # Create and save visualizations
        print("\nCreating visualizations...")
        radar_fig = plot_radar_chart(signature_2022, signature_2023)
        dist_fig = create_success_distribution_plot(df_2022, df_2023)
        # Create and save correlation heatmap
        corr_fig = create_correlation_heatmap(df_2022, df_2023)
        
        # Save visualizations
        radar_fig.write_html('audio_signatures_radar.html')
        dist_fig.write_html('success_metrics_distribution.html')
        if corr_fig is not None:
            corr_fig.write_html('success_factors_matrix.html')
            print(f"3. success_factors_matrix.html - Correlation heatmap of audio features and success metrics")
        
        # Print analysis summary
        print("\nVisualizations saved as:")
        print("1. audio_signatures_radar.html - Radar chart comparing audio features")
        print("2. success_metrics_distribution.html - Distribution of success metrics")
        
        # Print analysis summary
        print("\n=== AUDIO FEATURE EVOLUTION ===")
        evolution = signature_2023 - signature_2022
        for feature, change in evolution.items():
            print(f"{FEATURE_NAMES[feature]}: {change:+.1f}%")
    else:
        print("\nWarning: Could not generate visualizations - missing required data")
        if top_csi is None or top_csi.empty:
            print("- No 2022 data with CPS values found")
        if top_sdi is None or top_sdi.empty:
            print("- No 2023 data with VS values found")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    # Load, merge, and standardize the data
    merged_data = load_and_merge_spotify_data()
    merged_data, _ = standardize_integer_columns(merged_data)
    
    # Save the dataset
    output_file = 'spotify_merged_2022_2023.csv'
    merged_data.to_csv(output_file, index=False)
    print(f"\n✓ Dataset saved to {output_file}")
    
    # Display sample records
    print("\n" + "="*50)
    print("SAMPLE DATA")
    print("="*50)
    
    # 2022 sample
    print("\n2022 Top Tracks:")
    print(merged_data[merged_data['year'] == 2022]
          [['track_name', 'artist_name', 'peak_rank', 'weeks_on_chart']]
          .head(2).to_string(index=False))
    
    # 2023 sample
    print("\n2023 Top Tracks:")
    # Format streams with thousands separators
    sample_2023 = merged_data[merged_data['year'] == 2023][['track_name', 'artist_name', 'streams']].head(2).copy()
    sample_2023['streams'] = sample_2023['streams'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else '')
    print(sample_2023.to_string(index=False))
    
    # Run analysis on the merged data
    run_analysis(merged_data)