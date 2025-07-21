import pandas as pd
import numpy as np

# ============================================================================
# SPOTIFY DATASETS MERGE - COMPLETE IMPLEMENTATION
# ============================================================================

def load_and_merge_spotify_data():
    """
    Complete merging process for 2022 and 2023 Spotify datasets
    Returns a unified DataFrame ready for analysis
    """
    
    # Step 1: Load the datasets
    print("Loading datasets...")
    # Try different encodings for the files
    try:
        df_2022 = pd.read_csv('spotify_top_charts_22.csv')
    except UnicodeDecodeError:
        df_2022 = pd.read_csv('spotify_top_charts_22.csv', encoding='latin-1')
    
    try:
        df_2023 = pd.read_csv('spotify-2023.csv')
    except UnicodeDecodeError:
        df_2023 = pd.read_csv('spotify-2023.csv', encoding='latin-1')
    
    print(f"2022 dataset: {len(df_2022)} records")
    print(f"2023 dataset: {len(df_2023)} records")
    
    # Convert 2022 key format (0-11) to match 2023 format (C, C#, D, etc.)
    key_mapping = {
        0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
        6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
    }
    df_2022['key'] = df_2022['key'].map(key_mapping)
    print("Converted 2022 key format from numbers to musical notation")
    
    # Standardize tempo (round to nearest integer)
    df_2022['tempo'] = df_2022['tempo'].round().astype(int)
    
    # Standardize mode (0/1 to Minor/Major)
    mode_mapping = {1: 'Major', 0: 'Minor'}
    df_2022['mode'] = df_2022['mode'].map(mode_mapping)
    print("Standardized tempo (rounded) and mode (Minor/Major) for 2022 data")
    
    # Step 2: Standardize 2022 data
    print("Standardizing 2022 data...")
    df_2022_std = df_2022.copy()
    
    # Add year column
    df_2022_std['year'] = 2022
    
    # Standardize column names
    df_2022_std['artist_name'] = df_2022_std['artist_names']
    
    # Convert audio features from decimal (0.76) to percentage (76) to match 2023 format
    audio_features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                     'instrumentalness', 'liveness']
    
    for feature in audio_features:
        df_2022_std[f'{feature}_pct'] = (df_2022_std[feature] * 100).round().astype(int)
    
    # Add missing 2023 columns as null
    df_2022_std['streams'] = None
    df_2022_std['in_spotify_playlists'] = None
    df_2022_std['in_spotify_charts'] = None
    df_2022_std['valence_pct'] = None
    df_2022_std['released_year'] = None
    
    # Step 3: Standardize 2023 data  
    print("Standardizing 2023 data...")
    df_2023_std = df_2023.copy()
    
    # Add year column
    df_2023_std['year'] = 2023
    
    # Standardize column names
    df_2023_std['artist_name'] = df_2023_std['artist(s)_name']
    
    # Rename audio feature columns to match
    df_2023_std['danceability_pct'] = df_2023_std['danceability_%']
    df_2023_std['energy_pct'] = df_2023_std['energy_%']
    df_2023_std['speechiness_pct'] = df_2023_std['speechiness_%']
    df_2023_std['acousticness_pct'] = df_2023_std['acousticness_%']
    df_2023_std['instrumentalness_pct'] = df_2023_std['instrumentalness_%']
    df_2023_std['liveness_pct'] = df_2023_std['liveness_%']
    df_2023_std['valence_pct'] = df_2023_std['valence_%']
    
    # Standardize tempo column
    df_2023_std['tempo'] = df_2023_std['bpm']
    
    # Clean streams column (remove commas and convert to float)
    if df_2023_std['streams'].dtype == 'object':
        df_2023_std['streams'] = pd.to_numeric(df_2023_std['streams'].str.replace(',', ''), errors='coerce')
    
    # Add missing 2022 columns as null
    df_2023_std['peak_rank'] = None
    df_2023_std['weeks_on_chart'] = None
    df_2023_std['loudness'] = None
    df_2023_std['duration_ms'] = None
    df_2023_std['uri'] = None
    
    # Step 4: Select common columns for final merge
    common_columns = [
        'year', 'track_name', 'artist_name',
        'danceability_pct', 'energy_pct', 'speechiness_pct', 
        'acousticness_pct', 'instrumentalness_pct', 'liveness_pct', 'valence_pct',
        'key', 'mode', 'tempo',
        # 2022 specific
        'peak_rank', 'weeks_on_chart', 'loudness', 'duration_ms',
        # 2023 specific  
        'streams', 'in_spotify_playlists', 'in_spotify_charts', 'released_year'
    ]
    
    # Step 5: Create final merged dataset
    print("Creating unified dataset...")
    df_2022_final = df_2022_std[common_columns]
    df_2023_final = df_2023_std[common_columns]
    
    # Combine datasets
    df_merged = pd.concat([df_2022_final, df_2023_final], ignore_index=True)
    
    # Step 6: Data quality checks
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
    """Convert appropriate float columns to nullable Int64 type.
    
    Returns:
        DataFrame: DataFrame with standardized integer columns
        dict: Dictionary mapping column names to their original dtypes for reference
    """
    print("\nStandardizing integer columns...")
    
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
            
            # Print before/after info for the first column as example
            if col == 'peak_rank':
                print(f"Converting {col} from {df[col].dtype} to Int64")
                print(f"  Before: {df[col].head(3).to_list()}")
            
            # Convert to numeric first (handles any string representations)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert to nullable integer type
            df[col] = df[col].astype('Int64')
            
            if col == 'peak_rank':
                print(f"  After:  {df[col].head(3).to_list()}")
    
    return df, original_dtypes

def save_with_integer_types(df, filename):
    """Save DataFrame to CSV with proper handling of integer types.
    
    This function ensures that integer columns are properly formatted in the CSV
    without decimal points, even for nullable integer types.
    """
    # Make a copy to avoid modifying the original
    df_to_save = df.copy()
    
    # Convert Int64 columns to string representation to preserve integer format
    for col in df_to_save.select_dtypes(include=['Int64']).columns:
        # Convert to string, replacing <NA> with empty string
        df_to_save[col] = df_to_save[col].astype(str).replace('<NA>', '')
    
    # Save to CSV
    df_to_save.to_csv(filename, index=False)
    print(f"Saved to {filename} with proper integer formatting")

if __name__ == "__main__":
    # Load and merge the data
    merged_data = load_and_merge_spotify_data()
    
    # Standardize integer columns
    print("\n=== STANDARDIZING DATA TYPES ===")
    merged_data, original_dtypes = standardize_integer_columns(merged_data)
    
    # Print summary of type conversions
    print("\n=== TYPE CONVERSION SUMMARY ===")
    for col, orig_type in original_dtypes.items():
        new_type = merged_data[col].dtype
        print(f"{col}: {orig_type} -> {new_type}")
    
    # Save the merged dataset with proper integer handling
    output_file = 'spotify_merged_2022_2023.csv'
    save_with_integer_types(merged_data, output_file)
    
    # Verify the saved file
    print("\n=== VERIFYING SAVED FILE ===")
    df_check = pd.read_csv(output_file)
    print("\nSample values from saved file:")
    print(df_check[['peak_rank', 'weeks_on_chart', 'streams']].head(3).to_string())
    
    # Display sample records
    print("\n" + "="*50)
    print("SAMPLE MERGED DATA")
    print("="*50)
    print("\n2022 Sample:")
    print(merged_data[merged_data['year'] == 2022][['track_name', 'artist_name', 'peak_rank', 'weeks_on_chart', 'danceability_pct']].head(2))
    
    print("\n2023 Sample:")
    print(merged_data[merged_data['year'] == 2023][['track_name', 'artist_name', 'streams', 'in_spotify_playlists', 'danceability_pct']].head(2))