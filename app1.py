from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.losses import MeanSquaredError
import pandas as pd
from flask_cors import CORS

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
logging.basicConfig(level=logging.INFO)

# Load model and scalers for commodity prediction
def custom_lstm(*args, **kwargs):
    if 'time_major' in kwargs:
        del kwargs['time_major']
    return LSTM(*args, **kwargs)

try:
    model_commodity = load_model('lstm_model.h5', custom_objects={'LSTM': custom_lstm, 'mse': MeanSquaredError()})
    scaler_X_commodity = joblib.load('saved_data/scaler_X.pkl')
    scaler_y_commodity = joblib.load('saved_data/scaler_y.pkl')
    commodity_names = joblib.load('saved_data/commodity_names.pkl')
    data_commodity = pd.read_csv('dataset.csv')
    EXPECTED_FEATURE_COUNT = scaler_X_commodity.n_features_in_
except Exception as e:
    logging.error(f"Error loading commodity model or data: {e}")
    EXPECTED_FEATURE_COUNT = 0

# Load model and encoders for crop prediction
try:
    model_crop = joblib.load('crop_model.pkl')
    encoder_dict = joblib.load('encoders.pkl')
    data_crop = pd.read_csv('crop_pred.csv')
except Exception as e:
    logging.error(f"Error loading crop model or data: {e}")

categorical_columns = ['divisions', 'States']

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/predict_page", methods=['GET', 'POST'])
def predict_page():
    return render_template("index.html", commodities=commodity_names)

@app.route('/api/commodities')
def get_commodities():
    commodities = data_commodity['Commodity'].unique().tolist()
    return jsonify(commodities)

@app.route("/commodity_data/<commodity_name>")
def get_commodity_data(commodity_name):
    if commodity_name not in data_commodity['Commodity'].unique():
        return jsonify({"error": "Commodity not found."}), 404
    specific_data = data_commodity[data_commodity['Commodity'] == commodity_name]
    return jsonify({
        "dates": specific_data['Date'].astype(str).tolist(),
        "min_prices": specific_data['Minimum'].tolist(),
        "max_prices": specific_data['Maximum'].tolist()
    })

@app.route("/predict_commodity", methods=['POST'])
def predict_commodity():
    try:
        # Get the data from the form
        date = request.form['date']  # Format: 'YYYY-MM-DD'
        year, month, day = map(float, date.split('-'))
        commodity = request.form['commodity']

        if commodity not in commodity_names:
            return jsonify({"error": "Invalid commodity name."})

        # One-hot encode the commodity
        commodity_data = generate_one_hot_vector(commodity, commodity_names)

        # Padding if necessary
        feature_count = len(commodity_data) + 3
        if feature_count < EXPECTED_FEATURE_COUNT:
            commodity_data.extend([0] * (EXPECTED_FEATURE_COUNT - feature_count))

        input_data = np.array([year, month, day] + commodity_data).reshape(1, -1)
        input_scaled = scaler_X_commodity.transform(input_data)
        input_scaled = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))

        prediction = model_commodity.predict(input_scaled)
        prediction_original = scaler_y_commodity.inverse_transform(prediction).flatten()

        result = {
            'min_price': prediction_original[0],
            'max_price': prediction_original[1]
        }

        return render_template("result.html", result=result, commodity_name=commodity, date=date)
    except Exception as e:
        logging.error(f"Error in commodity prediction: {e}")
        return jsonify({"error": "An error occurred during prediction."})

def generate_one_hot_vector(input_category, all_possible_categories):
    return [1 if cat == input_category else 0 for cat in all_possible_categories]

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        user_input = request.get_json()

        # Encode categorical features in user input
        for col in categorical_columns:
            if col in user_input:
                try:
                    user_input[col] = encoder_dict[col].transform([user_input[col]])[0]
                except KeyError:
                    return jsonify({"error": f"Unseen label '{user_input[col]}' encountered for column '{col}'. Please provide a valid label."})

        user_df = pd.DataFrame([user_input])
        predicted_label = model_crop.predict(user_df)[0]
        return jsonify({"predicted_crop_label": predicted_label})
    
    except Exception as e:
        logging.error(f"Error in crop prediction: {e}")
        return jsonify({"error": "An error occurred during prediction."})

@app.route("/crop_prediction", methods=['GET', 'POST'])
def crop_prediction_page():
    return render_template("Crop_predection.html", commodities=commodity_names)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
