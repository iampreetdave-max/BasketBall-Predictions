"""
Basketball Match Predictor
Fetches today's and tomorrow's matches and generates predictions
"""

import os
import requests
import pandas as pd
import pickle
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')


class BasketballMatchPredictor:
    def __init__(self):
        # Get API key from environment variable
        self.api_key = os.environ.get('API_BASKETBALL_KEY')
        if not self.api_key:
            raise ValueError("API_BASKETBALL_KEY environment variable not set")
        
        self.base_url = "https://api-basketball.com/v1"
        self.headers = {
            'x-rapidapi-host': 'api-basketball.com',
            'x-rapidapi-key': self.api_key
        }
        self.league_id = 12  # NBA
        self.season = 2024
        self.model = None
        self.scaler = None
        
    def load_model(self, model_path='model/basketball_model.pkl'):
        """Load pre-trained model"""
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
            print("✓ Model loaded successfully")
            return True
        except FileNotFoundError:
            print(f"❌ Model file not found: {model_path}")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def make_api_request(self, endpoint, params=None):
        """Make API request with error handling"""
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
    
    def get_team_recent_games(self, team_id, limit=20):
        """Fetch recent games for a team"""
        params = {
            'league': self.league_id,
            'season': f"{self.season}-{self.season+1}",
            'team': team_id
        }
        
        data = self.make_api_request('games', params)
        games = data.get('response', []) if data else []
        
        # Filter completed games and sort by date
        completed = [g for g in games if g['scores']['home']['total'] is not None]
        completed.sort(key=lambda x: x['date'], reverse=True)
        
        return completed[:limit]
    
    def calculate_ewma(self, values, span=10):
        """Calculate Exponentially Weighted Moving Average"""
        if len(values) == 0:
            return 0
        return pd.Series(values).ewm(span=span, adjust=False).mean().iloc[-1]
    
    def extract_team_features(self, team_id, recent_games):
        """Extract rolling features for a team"""
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
        
        # EWMA features
        features['points_scored_ewma'] = self.calculate_ewma(points_scored, span=10)
        features['points_allowed_ewma'] = self.calculate_ewma(points_allowed, span=10)
        features['point_diff_ewma'] = features['points_scored_ewma'] - features['points_allowed_ewma']
        features['win_rate_ewma'] = self.calculate_ewma(wins, span=10)
        
        # Recent form
        features['recent_form'] = np.mean(wins[-5:])
        features['points_scored_recent'] = np.mean(points_scored[-5:])
        features['points_allowed_recent'] = np.mean(points_allowed[-5:])
        
        # Season averages
        features['points_scored_avg'] = np.mean(points_scored)
        features['points_allowed_avg'] = np.mean(points_allowed)
        features['win_rate'] = np.mean(wins)
        
        # Consistency
        features['scoring_std'] = np.std(points_scored)
        features['defense_std'] = np.std(points_allowed)
        
        return features
    
    def prepare_prediction_features(self, home_team_id, away_team_id):
        """Prepare features for prediction"""
        # Get recent games
        home_games = self.get_team_recent_games(home_team_id)
        away_games = self.get_team_recent_games(away_team_id)
        
        # Extract features
        home_features = self.extract_team_features(home_team_id, home_games)
        away_features = self.extract_team_features(away_team_id, away_games)
        
        if home_features is None or away_features is None:
            return None
        
        # Create differential features
        matchup_features = {}
        for key in home_features.keys():
            matchup_features[f'{key}_diff'] = home_features[key] - away_features[key]
        
        matchup_features['home_court'] = 1
        
        return matchup_features
    
    def predict_match(self, home_team_id, away_team_id, home_team_name, away_team_name):
        """Predict match outcome"""
        if self.model is None:
            return None
        
        # Prepare features
        features = self.prepare_prediction_features(home_team_id, away_team_id)
        
        if features is None:
            return {
                'home_team': home_team_name,
                'away_team': away_team_name,
                'prediction': 'Insufficient Data',
                'home_win_prob': 0,
                'away_win_prob': 0,
                'confidence': 0
            }
        
        # Get feature names in correct order
        feature_cols = [k for k in features.keys() if k not in ['outcome', 'date', 'home_team_id', 'away_team_id']]
        X = pd.DataFrame([features])[feature_cols]
        X = X.fillna(0)
        
        # Predict
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        result = {
            'home_team': home_team_name,
            'away_team': away_team_name,
            'prediction': 'Home Win' if prediction == 1 else 'Away Win',
            'home_win_prob': round(probabilities[1] * 100, 2),
            'away_win_prob': round(probabilities[0] * 100, 2),
            'confidence': round(max(probabilities) * 100, 2)
        }
        
        return result
    
    def get_upcoming_matches(self):
        """Get today's and tomorrow's matches"""
        today = datetime.now()
        tomorrow = today + timedelta(days=1)
        
        today_str = today.strftime('%Y-%m-%d')
        tomorrow_str = tomorrow.strftime('%Y-%m-%d')
        
        print(f"\nFetching matches for {today_str} and {tomorrow_str}...")
        
        today_games = self.get_games_for_date(today_str)
        tomorrow_games = self.get_games_for_date(tomorrow_str)
        
        all_games = []
        
        # Process today's games
        for game in today_games:
            all_games.append({
                'date': today_str,
                'game': game
            })
        
        # Process tomorrow's games
        for game in tomorrow_games:
            all_games.append({
                'date': tomorrow_str,
                'game': game
            })
        
        print(f"✓ Found {len(all_games)} upcoming matches")
        return all_games
    
    def predict_upcoming_matches(self):
        """Main function to predict upcoming matches"""
        # Load model
        if not self.load_model():
            print("Cannot proceed without model!")
            return None
        
        # Get matches
        matches = self.get_upcoming_matches()
        
        if not matches:
            print("No upcoming matches found")
            return pd.DataFrame()
        
        # Make predictions
        predictions = []
        
        for match_data in matches:
            game = match_data['game']
            date = match_data['date']
            
            home_id = game['teams']['home']['id']
            away_id = game['teams']['away']['id']
            home_name = game['teams']['home']['name']
            away_name = game['teams']['away']['name']
            
            print(f"\nPredicting: {home_name} vs {away_name}")
            
            result = self.predict_match(home_id, away_id, home_name, away_name)
            
            if result:
                result['date'] = date
                result['game_time'] = game.get('time', 'TBD')
                predictions.append(result)
                
                print(f"  Prediction: {result['prediction']} ({result['confidence']}%)")
        
        # Create DataFrame
        df = pd.DataFrame(predictions)
        
        # Reorder columns
        if not df.empty:
            df = df[['date', 'game_time', 'home_team', 'away_team', 
                    'prediction', 'home_win_prob', 'away_win_prob', 'confidence']]
        
        return df


def main():
    """Main execution"""
    print("="*60)
    print("Basketball Match Predictor")
    print("="*60)
    
    try:
        predictor = BasketballMatchPredictor()
        predictions_df = predictor.predict_upcoming_matches()
        
        if predictions_df is not None and not predictions_df.empty:
            # Save to CSV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'predictions/predictions_{timestamp}.csv'
            
            # Create predictions directory if it doesn't exist
            os.makedirs('predictions', exist_ok=True)
            
            predictions_df.to_csv(filename, index=False)
            print(f"\n✓ Predictions saved to {filename}")
            
            # Also save as latest.csv for easy access
            predictions_df.to_csv('predictions/latest.csv', index=False)
            print("✓ Latest predictions saved to predictions/latest.csv")
            
            # Display predictions
            print("\n" + "="*60)
            print("PREDICTIONS")
            print("="*60)
            print(predictions_df.to_string(index=False))
            
        else:
            print("\nNo predictions generated")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
