import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('clean_matches.csv')

# Date features
df['date'] = pd.to_datetime(df['date'])
df['Year'] = df['date'].dt.year
df['Month'] = df['date'].dt.month
df['Day'] = df['date'].dt.day

# Drop unnecessary columns
df = df.drop(columns=['match_api_id', 'date'])

# Encode teams
le_home = LabelEncoder()
le_away = LabelEncoder()
df['home_team'] = le_home.fit_transform(df['home_team'])
df['away_team'] = le_away.fit_transform(df['away_team'])

# Encode result
df['result'] = df['result'].map({'H': 0, 'D': 1, 'A': 2})

# Features and target
X = df.drop(columns=['result'])
y = df['result']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model with parameters to reduce overfitting
model = RandomForestClassifier(
    n_estimators=100,  # Number of trees in the forest
    max_depth=10,      # Limit the depth of trees
    min_samples_split=10,  # Minimum samples required to split an internal node
    min_samples_leaf=5,  # Minimum samples required to be at a leaf node
    random_state=42
)

# Perform cross-validation to evaluate the model's generalization
cv_scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean()}")

# Train the model on the full training data
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("📋 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', 
            xticklabels=['Home Win', 'Draw', 'Away Win'],
            yticklabels=['Home Win', 'Draw', 'Away Win'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Count plot of result
sns.countplot(x='result', data=df, palette='Set2')
plt.xticks([0, 1, 2], ['Home Win', 'Draw', 'Away Win'])
plt.title("Distribution of Match Results")
plt.xlabel("Result")
plt.ylabel("Count")
plt.show()

# Save the trained model and encoders
joblib.dump(model, 'football_match_predictor_rf_model.pkl')
joblib.dump(le_home, 'home_encoder.pkl')
joblib.dump(le_away, 'away_encoder.pkl')

print("✅ Model and encoders saved successfully.")
