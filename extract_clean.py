import sqlite3
import pandas as pd

# Connect to SQLite
conn = sqlite3.connect('database.sqlite')

# Show available tables (optional)
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
print("Tables:", tables)

# Load Match data
query = """
SELECT 
    m.match_api_id, 
    m.date, 
    ht.team_long_name AS home_team,
    at.team_long_name AS away_team,
    m.home_team_goal, 
    m.away_team_goal
FROM Match m
JOIN Team ht ON m.home_team_api_id = ht.team_api_id
JOIN Team at ON m.away_team_api_id = at.team_api_id
WHERE m.home_team_goal IS NOT NULL AND m.away_team_goal IS NOT NULL
"""
df = pd.read_sql(query, conn)

# Add match result column
def match_result(row):
    if row['home_team_goal'] > row['away_team_goal']:
        return 'H'
    elif row['home_team_goal'] < row['away_team_goal']:
        return 'A'
    else:
        return 'D'

df['result'] = df.apply(match_result, axis=1)

# Save clean data
df.to_csv('clean_matches.csv', index=False)
print("✅ Clean data saved as clean_matches.csv")
