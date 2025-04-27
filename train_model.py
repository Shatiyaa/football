import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('clean_matches.csv')

# Encode categorical
le_home = LabelEncoder()
le_away = LabelEncoder()
df['home_team'] = le_home.fit_transform(df['home_team'])
df['away_team'] = le_away.fit_transform(df['away_team'])
df['result'] = df['result'].map({'H': 0, 'D': 1, 'A': 2})

# Only meaningful features
X = df[['home_team', 'away_team']]
y = df['result']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'football_match_predictor_rf_model.pkl')
joblib.dump(le_home, 'home_encoder.pkl')
joblib.dump(le_away, 'away_encoder.pkl')

print("✅ Model and encoders saved successfully.")
