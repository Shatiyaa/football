from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

model = joblib.load('football_match_predictor_rf_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array([[int(data['home_team']),
                              int(data['away_team']),
                              float(data['home_team_recent_points']),
                              float(data['away_team_recent_points']),
                              int(data['year']),
                              int(data['month']),
                              int(data['day'])]])

        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)[0]

        labels = ['Home Win (H)', 'Draw (D)', 'Away Win (A)']
        fig, ax = plt.subplots()
        bars = ax.bar(labels, prediction_proba, color=['#4caf50', '#ffeb3b', '#f44336'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probabilities')

        for bar in bars:
            ax.annotate(f'{bar.get_height():.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', fontsize=9)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)

        return jsonify({
            'prediction': int(prediction[0]),
            'image': image_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
