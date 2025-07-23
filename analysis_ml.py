import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ML and preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance

# Visualization and analysis
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap

# Statistical analysis
from scipy import stats
from scipy.stats import chi2_contingency

class SpotifyHitPredictor:
    """
    A comprehensive ML pipeline for predicting hit songs based on audio features
    and historical success patterns.
    """
    
    def __init__(self, data_path='spotify_merged_2022_2023.csv'):
        """Initialize the predictor with data loading and preprocessing."""
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load data and create target variable for hit prediction."""
        print("📊 Loading and preparing data...")
        
        # Load the merged dataset
        self.df = pd.read_csv(self.data_path)
        
        # Calculate unified success scores
        self.df['unified_success_score'] = self._calculate_unified_success_score()
        
        # Create hit classification (top 20% = hit, bottom 80% = not hit)
        success_threshold = self.df['unified_success_score'].quantile(0.8)
        self.df['is_hit'] = (self.df['unified_success_score'] >= success_threshold).astype(int)
        
        # Feature engineering
        self.df = self._engineer_features()
        
        print(f"✅ Data loaded: {len(self.df)} tracks")
        print(f"🎯 Hit rate: {self.df['is_hit'].mean():.1%}")
        print(f"📈 Success score range: {self.df['unified_success_score'].min():.2f} - {self.df['unified_success_score'].max():.2f}")
        
        return self.df
    
    def _calculate_unified_success_score(self):
        """Create a unified success score across both years using percentile ranking."""
        unified_scores = np.zeros(len(self.df))
        
        # 2022: Use Chart Performance Score percentile
        mask_2022 = self.df['year'] == 2022
        if mask_2022.sum() > 0:
            cps_scores = (self.df.loc[mask_2022, 'weeks_on_chart'] * 
                         (101 - self.df.loc[mask_2022, 'peak_rank']) / 100)
            unified_scores[mask_2022] = stats.rankdata(cps_scores) / len(cps_scores)
        
        # 2023: Use Virality Score percentile
        mask_2023 = self.df['year'] == 2023
        if mask_2023.sum() > 0:
            # Handle missing values in streams and playlists
            streams = pd.to_numeric(self.df.loc[mask_2023, 'streams'], errors='coerce').fillna(0)
            playlists = pd.to_numeric(self.df.loc[mask_2023, 'in_spotify_playlists'], errors='coerce').fillna(0)
            
            vs_scores = (streams / 1_000_000) * np.sqrt(playlists / 1000)
            unified_scores[mask_2023] = stats.rankdata(vs_scores) / len(vs_scores)
        
        return unified_scores
    
    def _engineer_features(self):
        """Create additional features for better prediction."""
        df = self.df.copy()
        
        # Audio feature combinations
        df['energy_dance_ratio'] = df['energy_pct'] / (df['danceability_pct'] + 1)
        df['acoustic_energy_balance'] = df['acousticness_pct'] - df['energy_pct']
        df['vocal_content'] = df['speechiness_pct'] + (100 - df['instrumentalness_pct'])
        
        # Tempo categories
        df['tempo_category'] = pd.cut(df['tempo'], 
                                    bins=[0, 90, 120, 140, 300], 
                                    labels=['Slow', 'Medium', 'Fast', 'Very_Fast'])
        
        # Key popularity (based on common pop music keys)
        popular_keys = ['C', 'G', 'D', 'A', 'F']
        df['popular_key'] = df['key'].isin(popular_keys).astype(int)
        
        # Mode encoding
        df['is_major'] = (df['mode'] == 'Major').astype(int)
        
        # Year-specific features
        df['is_2023'] = (df['year'] == 2023).astype(int)
        
        # Audio feature diversity (how "unique" the sound signature is)
        audio_features = ['danceability_pct', 'energy_pct', 'speechiness_pct', 
                         'acousticness_pct', 'instrumentalness_pct', 'liveness_pct']
        df['feature_diversity'] = df[audio_features].std(axis=1)
        
        return df
    
    def prepare_features_target(self):
        """Prepare feature matrix and target variable for ML models."""
        
        # Select features for modeling
        audio_features = ['danceability_pct', 'energy_pct', 'speechiness_pct', 
                         'acousticness_pct', 'instrumentalness_pct', 'liveness_pct']
        
        engineered_features = ['energy_dance_ratio', 'acoustic_energy_balance', 
                              'vocal_content', 'feature_diversity', 'popular_key', 
                              'is_major', 'is_2023']
        
        # Include valence for 2023 data
        if 'valence_pct' in self.df.columns:
            valence_feature = self.df['valence_pct'].fillna(self.df['valence_pct'].mean())
            self.df['valence_pct_filled'] = valence_feature
            audio_features.append('valence_pct_filled')
        
        # Combine all features
        feature_columns = audio_features + engineered_features + ['tempo']
        
        # Prepare feature matrix
        X = self.df[feature_columns].copy()
        
        # Handle categorical variables
        if 'tempo_category' in self.df.columns:
            # Add tempo category as dummy variables
            tempo_dummies = pd.get_dummies(self.df['tempo_category'], prefix='tempo')
            X = pd.concat([X, tempo_dummies], axis=1)
        
        # Fill any remaining missing values
        X = X.fillna(X.mean())
        
        # Target variable
        y = self.df['is_hit']
        
        print(f"🔧 Feature matrix shape: {X.shape}")
        print(f"📊 Features: {list(X.columns)}")
        
        return X, y
    
    def train_models(self, test_size=0.2, random_state=42):
        """Train multiple ML models and compare performance."""
        print("\n🤖 Training ML models...")
        
        # Prepare data
        X, y = self.prepare_features_target()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=random_state, 
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=6, random_state=random_state
            ),
            'Logistic Regression': LogisticRegression(
                random_state=random_state, class_weight='balanced', max_iter=1000
            ),
            'SVM': SVC(
                probability=True, random_state=random_state, class_weight='balanced'
            )
        }
        
        # Train and evaluate models
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        
        for name, model in models.items():
            print(f"\n🎯 Training {name}...")
            
            # Use scaled data for SVM and Logistic Regression
            if name in ['SVM', 'Logistic Regression']:
                X_train_model, X_test_model = X_train_scaled, X_test_scaled
            else:
                X_train_model, X_test_model = X_train, X_test
            
            # Train model
            model.fit(X_train_model, y_train)
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train_model, y_train, cv=cv, scoring='roc_auc')
            
            # Test predictions
            y_pred = model.predict(X_test_model)
            y_pred_proba = model.predict_proba(X_test_model)[:, 1]
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_auc': roc_auc_score(y_test, y_pred_proba),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"✅ CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"📊 Test AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
        
        # Store test data for analysis
        self.X_test = X_test
        self.y_test = y_test
        
        return self.results
    
    def analyze_feature_importance(self):
        """Analyze and visualize feature importance across models."""
        print("\n🔍 Analyzing feature importance...")
        
        # Get best model (highest test AUC)
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['test_auc'])
        best_model = self.models[best_model_name]
        
        print(f"🏆 Best model: {best_model_name} (AUC: {self.results[best_model_name]['test_auc']:.3f})")
        
        # Feature importance for tree-based models
        if hasattr(best_model, 'feature_importances_'):
            feature_names = self.X_test.columns
            importances = best_model.feature_importances_
            
            self.feature_importance[best_model_name] = dict(zip(feature_names, importances))
            
            # Sort features by importance
            sorted_features = sorted(self.feature_importance[best_model_name].items(), 
                                   key=lambda x: x[1], reverse=True)
            
            print("\n📈 Top 10 Most Important Features:")
            for feature, importance in sorted_features[:10]:
                print(f"  {feature}: {importance:.3f}")
        
        return self.feature_importance
    
    def create_visualizations(self):
        """Create comprehensive visualizations of model performance and insights."""
        print("\n📊 Creating visualizations...")
        
        # 1. Model Comparison
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance Comparison', 'ROC Curves', 
                           'Feature Importance', 'Success Score Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Model performance comparison
        model_names = list(self.results.keys())
        cv_means = [self.results[name]['cv_mean'] for name in model_names]
        test_aucs = [self.results[name]['test_auc'] for name in model_names]
        
        fig.add_trace(
            go.Bar(name='CV AUC', x=model_names, y=cv_means, marker_color='lightblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Test AUC', x=model_names, y=test_aucs, marker_color='darkblue'),
            row=1, col=1
        )
        
        # ROC Curves
        for name in model_names:
            fpr, tpr, _ = roc_curve(self.results[name]['y_test'], 
                                   self.results[name]['y_pred_proba'])
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, name=f'{name} (AUC={self.results[name]["test_auc"]:.3f})',
                          mode='lines'),
                row=1, col=2
            )
        
        # Add diagonal line for ROC
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                      line=dict(dash='dash', color='gray')),
            row=1, col=2
        )
        
        # Feature importance (for best model)
        if self.feature_importance:
            best_model_name = max(self.results.keys(), 
                                 key=lambda x: self.results[x]['test_auc'])
            if best_model_name in self.feature_importance:
                features = list(self.feature_importance[best_model_name].keys())[:10]
                importances = [self.feature_importance[best_model_name][f] for f in features]
                
                fig.add_trace(
                    go.Bar(x=importances, y=features, orientation='h', 
                          name='Feature Importance', marker_color='green'),
                    row=2, col=1
                )
        
        # Success score distribution
        fig.add_trace(
            go.Histogram(x=self.df['unified_success_score'], nbinsx=30,
                        name='Success Score Distribution', marker_color='purple'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Spotify Hit Prediction Model Analysis")
        fig.update_xaxes(title_text="Models", row=1, col=1)
        fig.update_yaxes(title_text="AUC Score", row=1, col=1)
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)
        fig.update_xaxes(title_text="Importance", row=2, col=1)
        fig.update_xaxes(title_text="Success Score", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        fig.show()
        
        return fig
    
    def predict_hit_probability(self, track_features):
        """Predict hit probability for a new track."""
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['test_auc'])
        best_model = self.models[best_model_name]
        
        # Prepare features
        if isinstance(track_features, dict):
            # Convert dict to DataFrame
            track_df = pd.DataFrame([track_features])
        else:
            track_df = track_features.copy()
        
        # Apply same preprocessing
        if best_model_name in ['SVM', 'Logistic Regression']:
            track_features_scaled = self.scalers['standard'].transform(track_df)
            hit_probability = best_model.predict_proba(track_features_scaled)[0, 1]
        else:
            hit_probability = best_model.predict_proba(track_df)[0, 1]
        
        return hit_probability
    
    def generate_insights(self):
        """Generate actionable insights from the model analysis."""
        print("\n💡 Generating Insights...")
        
        insights = {
            'model_performance': {},
            'feature_insights': {},
            'business_recommendations': []
        }
        
        # Model performance insights
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['test_auc'])
        best_auc = self.results[best_model_name]['test_auc']
        
        insights['model_performance'] = {
            'best_model': best_model_name,
            'best_auc': best_auc,
            'performance_tier': 'Excellent' if best_auc > 0.8 else 'Good' if best_auc > 0.7 else 'Fair'
        }
        
        # Feature insights
        if self.feature_importance and best_model_name in self.feature_importance:
            top_features = sorted(self.feature_importance[best_model_name].items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            insights['feature_insights'] = dict(top_features)
        
        # Business recommendations
        recommendations = [
            f"🎯 The {best_model_name} model achieves {best_auc:.1%} accuracy in predicting hits",
            "🎵 Focus on audio features that matter most for hit prediction",
            "📊 Use this model to evaluate new releases before major marketing spend",
            "🔄 Regularly retrain the model with new data to maintain accuracy"
        ]
        
        if self.feature_importance:
            top_feature = max(self.feature_importance[best_model_name].items(), 
                            key=lambda x: x[1])[0]
            recommendations.append(f"⭐ '{top_feature}' is the most predictive feature - optimize for this")
        
        insights['business_recommendations'] = recommendations
        
        # Print insights
        print(f"\n🏆 Best Model: {best_model_name} (AUC: {best_auc:.3f})")
        print("\n📈 Top Predictive Features:")
        if insights['feature_insights']:
            for feature, importance in list(insights['feature_insights'].items())[:5]:
                print(f"  • {feature}: {importance:.3f}")
        
        print("\n💼 Business Recommendations:")
        for rec in recommendations:
            print(f"  {rec}")
        
        return insights

def main():
    """Main execution function."""
    print("🎵 Spotify Hit Prediction Model")
    print("=" * 50)
    
    # Initialize predictor
    predictor = SpotifyHitPredictor()
    
    # Load and prepare data
    df = predictor.load_and_prepare_data()
    
    # Train models
    results = predictor.train_models()
    
    # Analyze feature importance
    feature_importance = predictor.analyze_feature_importance()
    
    # Create visualizations
    fig = predictor.create_visualizations()
    
    # Generate insights
    insights = predictor.generate_insights()
    
    print("\n✅ Analysis complete!")
    print("🎯 Your ML model is ready to predict hit songs!")
    
    return predictor, results, insights

if __name__ == "__main__":
    predictor, results, insights = main()