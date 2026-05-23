from flask import Flask, request, jsonify
import joblib
import pandas as pd

#Load trained model
model = joblib.load('dss_model.pkl')

#Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
        app.run(debug=True)
    