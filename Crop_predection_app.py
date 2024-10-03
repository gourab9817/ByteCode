from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib  # To load the model
from flask_cors import CORS

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Load model and encoders
model = joblib.load('crop_model.pkl')
encoder_dict = joblib.load('encoders.pkl')

# Load dataset for structure (optional, if needed)
data = pd.read_csv(r'D:\freshForecast-main\freshForecast-main\flask_app\crop_pred.csv')

categorical_columns = ['divisions', 'States']

@app.route('/')
def home():
    return render_template('Crop_predection.html')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        # Get user input from the request
        user_input = request.get_json()

        # Encode categorical features in user input
        for col in categorical_columns:
            try:
                user_input[col] = encoder_dict[col].transform([user_input[col]])[0]
            except KeyError:
                return jsonify({"error": f"Unseen label '{user_input[col]}' encountered for column '{col}'. Please provide a valid label."})

        # Convert the input to DataFrame
        user_df = pd.DataFrame([user_input])

        # Make prediction using pre-trained model
        predicted_label = model.predict(user_df)[0]

        return jsonify({"predicted_crop_label": predicted_label})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
