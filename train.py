"""
Train Basketball Prediction Model
One-time script to train and save the model
"""

import os
import requests
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api-basketball.com/v1"
        self.headers = {
            'x-rapidapi-host': 'api-basketball.com',
            'x-rapidapi-key': api_key
        }
        self.league_id = 12  # NBA (DO NOT CHANGE - configured for NBA only)
        self.season = 2024  # Current NBA season
        self.model = None
        self.scaler = StandardScaler()
        
    def make_api_request(self, endpoint, params=None):
        """Make API request"""
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API Error: {e}")
            return None
    
    def get_games_for_date(self, date):
        """Fetch games for a specific date"""
        params = {
            'league': self.league_id,
            'season': f"{self.season}-{self.season+1}",
            'date': date
        }
        data = self.make_api_request('games', params)
        return data.get('response', []) if data else []
    
    def get_team_recent_games(self, team_id, before_date=None):
        """Fetch recent games for a team"""
        params = {
            'league': self.league_id,
            'season': f"{self.season}-{self.season+1}",
            'team': team_id
        }
        data = self.make_api_request('games', params)
        games = data.get('response', []) if data else []
        
        # Filter completed games
        completed = [g for g in games if g['scores']['home']['total'] is not None]
        
        # Filter by date if specified
        if before_date:
            completed = [g for g in completed if g['date'] < before_date]
        
        completed.sort(key=lambda x: x['date'], reverse=True)
        return completed[:20]
    
    def calculate_ewma(self, values, span=10):
        """Calculate EWMA"""
        if len(values) == 0:
            return 0
        return pd.Series(values).ewm(span=span, adjust=False).mean().iloc[-1]
    
    def extract_team_features(self, team_id, recent_games):
        """Extract team features"""
        if not recent_games or len(recent_games) < 5:
            return None
        
        features = {}
        points_scored = []
        points_allowed = []
        wins = []
        
        for game in recent_games:
            is_home = game['teams']['home']['id'] == team_id
            team_key = 'home' if is_home else 'away'
            opp_key = 'away' if is_home else 'home'
            
            if game['scores'][team_key]['total'] is not None:
                points_scored.append(game['scores'][team_key]['total'])
                points_allowed.append(game['scores'][opp_key]['total'])
                won = game['scores'][team_key]['total'] > game['scores'][opp_key]['total']
                wins.append(1 if won else 0)
        
        if len(points_scored) < 5:
            return None
        
        features['points_scored_ewma'] = self.calculate_ewma(points_scored, span=10)
        features['points_allowed_ewma'] = self.calculate_ewma(points_allowed, span=10)
        features['point_diff_ewma'] = features['points_scored_ewma'] - features['points_allowed_ewma']
        features['win_rate_ewma'] = self.calculate_ewma(wins, span=10)
        features['recent_form'] = np.mean(wins[-5:])
        features['points_scored_recent'] = np.mean(points_scored[-5:])
        features['points_allowed_recent'] = np.mean(points_allowed[-5:])
        features['points_scored_avg'] = np.mean(points_scored)
        features['points_allowed_avg'] = np.mean(points_allowed)
        features['win_rate'] = np.mean(wins)
        features['scoring_std'] = np.std(points_scored)
        features['defense_std'] = np.std(points_allowed)
        
        return features
    
    def collect_training_data(self, start_date, end_date, sample_days=30):
        """Collect training data by sampling dates"""
        print(f"\nCollecting training data from {start_date} to {end_date}")
        print(f"Sampling {sample_days} days to save API calls...")
        
        training_data = []
        
        # Parse dates
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        total_days = (end - start).days
        
        # Sample dates evenly across the range
        if total_days > sample_days:
            sample_interval = total_days // sample_days
            dates_to_sample = [start + timedelta(days=i*sample_interval) for i in range(sample_days)]
        else:
            dates_to_sample = [start + timedelta(days=i) for i in range(total_days + 1)]
        
        for date_obj in dates_to_sample:
            date_str = date_obj.strftime('%Y-%m-%d')
            print(f"Processing {date_str}...", end=' ')
            
            games = self.get_games_for_date(date_str)
            
            games_processed = 0
            for game in games:
                if game['scores']['home']['total'] is None:
                    continue
                
                home_id = game['teams']['home']['id']
                away_id = game['teams']['away']['id']
                
                # Get features before this game
                home_games = self.get_team_recent_games(home_id, before_date=date_str)
                away_games = self.get_team_recent_games(away_id, before_date=date_str)
                
                home_features = self.extract_team_features(home_id, home_games)
                away_features = self.extract_team_features(away_id, away_games)
                
                if home_features is None or away_features is None:
                    continue
                
                # Create differential features
                matchup_features = {}
                for key in home_features.keys():
                    matchup_features[f'{key}_diff'] = home_features[key] - away_features[key]
                matchup_features['home_court'] = 1
                
                # Outcome
                home_score = game['scores']['home']['total']
                away_score = game['scores']['away']['total']
                matchup_features['outcome'] = 1 if home_score > away_score else 0
                
                training_data.append(matchup_features)
                games_processed += 1
            
            print(f"{games_processed} games")
        
        df = pd.DataFrame(training_data)
        print(f"\n✓ Collected {len(df)} games for training")
        return df
    
    def train(self, training_df):
        """Train the model"""
        print("\nTraining model...")
        
        # Separate features and target
        feature_cols = [col for col in training_df.columns if col != 'outcome']
        X = training_df[feature_cols]
        y = training_df['outcome']
        
        X = X.fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"\n✓ Training Accuracy: {train_acc:.2%}")
        print(f"✓ Testing Accuracy: {test_acc:.2%}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"✓ Cross-Validation: {cv_scores.mean():.2%} (±{cv_scores.std():.2%})")
        
        print("\nClassification Report:")
        print(classification_report(y_test, test_pred, target_names=['Away Win', 'Home Win']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return self.model
    
    def save_model(self, filepath='model/basketball_model.pkl'):
        """Save model and scaler"""
        os.makedirs('model', exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'league_id': self.league_id,
                'season': self.season
            }, f)
        
        print(f"\n✓ Model saved to {filepath}")


def main():
    """Main training function"""
    print("="*60)
    print("Basketball Model Training")
    print("="*60)
    
    # Get API key
    api_key = os.environ.get('API_BASKETBALL_KEY')
    if not api_key:
        api_key = input("\nEnter your API-Basketball key: ").strip()
        if not api_key:
            print("❌ API key required!")
            return
    
    # Initialize trainer
    trainer = ModelTrainer(api_key)
    
    # Collect training data
    # Use sampling to reduce API calls
    # Adjust dates and sample_days based on your needs
    start_date = "2024-10-22"  # Season start
    end_date = "2024-12-31"    # Training end
    sample_days = 30  # Sample 30 days to save API calls
    
    training_df = trainer.collect_training_data(start_date, end_date, sample_days)
    
    if len(training_df) < 20:
        print(f"\n❌ Insufficient training data: {len(training_df)} games")
        print("Try extending the date range or increasing sample_days")
        return
    
    # Save training data
    os.makedirs('data', exist_ok=True)
    training_df.to_csv('data/training_data.csv', index=False)
    print(f"✓ Training data saved to data/training_data.csv")
    
    # Train model
    trainer.train(training_df)
    
    # Save model
    trainer.save_model()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nYou can now use predict_matches.py to make predictions")


if __name__ == "__main__":
    main()
