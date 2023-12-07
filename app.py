from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import datetime as dt


app = Flask(__name__)

model_filename = 'best_estimator_RF.pkl'
scaler_filename = 'scaler.pkl'

loaded_model = pickle.load(open(model_filename,'rb'))
loaded_scaler = pickle.load(open(scaler_filename,'rb'))

def map_zip_code(zip_code): 
    zip_code_stats = pd.read_csv("stats_by_zipcode.csv", index_col = "zipcode")
    zip_code_data = zip_code_stats.loc[zip_code]
    return list(zip_code_data)[:-1]



@app.route('/predict', methods=['GET'])
def predict():
    try:
        data = []
        # Extract query parameters from the URL
        bed = float(request.args.get('bed'))
        bath = float(request.args.get('bath'))
        acre_lot = float(request.args.get('acre_lot'))
        house_size = float(request.args.get('house_size'))
        zip_code = float(request.args.get('zip_code'))

        data.append(bed)
        data.append(bath)
        data.append(acre_lot)
        data.append(house_size)
        data = data + map_zip_code(zip_code)

        data.append(2023)
        data.append(8)
        data.append(-5)

        # Preprocess input features with the scaler
        scaled_features = loaded_scaler.transform([data])
        prediction = loaded_model.predict(scaled_features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
