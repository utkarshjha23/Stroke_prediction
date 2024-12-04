from flask import Flask, request, jsonify
import pandas as pd
from joblib import load
from flask_cors import CORS
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Load the model
model = load("./stroke_model.joblib")
pipeline = load("./stroke_pipeline2.joblib")

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.json
        print("DATA: ", data)
        df = pd.DataFrame([data])
        print(df.head())
        prediction = pipeline.predict(df)[0]
        print(f"Prediction: {int(prediction)}")
        prediction = int(prediction)
        
        return jsonify({"stroke": "Stroke" if prediction > 0 else "No Stroke"}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "Welcome to Stroke Prediction API"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
