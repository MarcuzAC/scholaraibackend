from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import numpy as np
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


MONGODB_URI = os.getenv("MongoDB_URI")

app = Flask(__name__)
CORS(app)

# MongoDB configuration
client = MongoClient(MONGODB_URI)
db = client['scholarai']
collection = db['predictions']

# Check MongoDB connection
try:
    db.list_collection_names() 
    print("Successfully connected to MongoDB.")
except Exception as e:
    print("MongoDB connection failed:", e)

# Load the pre-trained model
model_path = 'stem_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

# Function to generate study recommendations
def generate_recommendations(input_data):
    recommendations = []
    
    if input_data.get('mathscore', 0) < 50:
        recommendations.append("Improve your Math skills by practicing more problems.")
    if input_data.get('physicsscore', 0) < 50:
        recommendations.append("Enhance your Physics understanding through additional tutorials.")
    if input_data.get('chemscore', 0) < 50:
        recommendations.append("Focus on Chemistry concepts and seek help if needed.")
    if input_data.get('absences', 0) > 10:
        recommendations.append("Reduce your absences to stay up-to-date with the coursework.")
    if input_data.get('studytimeweekly', 0) < 5:
        recommendations.append("Increase your weekly study time to improve your scores.")
    if not recommendations:
        recommendations.append("Great job! Keep up the good work.")
    return recommendations

# Route to handle root requests to reduce 404 errors
@app.route('/')
def index():
    return jsonify({'status': 'Application is running'})

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded properly'}), 500

    input_data = request.get_json(force=True)
    print("Received input data:", input_data)

    try:
        math_score = int(input_data['mathscore'])
        physics_score = int(input_data['physicsscore'])
        chem_score = int(input_data['chemscore'])
        average = (math_score + physics_score + chem_score) // 3

        edu_mapping = {
            "College": 1,
            "Secondary School": 2,
            "Primary School": 3,
        }
        lunch_mapping = {
            "Standard": 1,
            "Reduced": 0
        }
        testprep_mapping = {
            "Completed": 0,
            "Not Completed": 1
        }
        gender_mapping = {
            "Male": 0,
            "Female": 1
        }

        features = [
            gender_mapping.get(input_data['gender'], 0),           
            edu_mapping.get(input_data['parentedu'], 3),           
            lunch_mapping.get(input_data['lunch'], 1),             
            testprep_mapping.get(input_data['testprep'], 1),       
            math_score,
            physics_score,
            chem_score,
            int(input_data.get('studytimeweekly', 5)),             
            int(input_data.get('absences', 0)),                    
            average
        ]
    except (KeyError, ValueError, TypeError) as e:
        print("Error in input data:", e)
        return jsonify({'error': 'Invalid input data', 'details': str(e)}), 400

    try:
        model_input = np.array([features])
        prediction = model.predict(model_input)
        predicted_class = int(np.argmax(prediction[0]))
        print("Prediction result:", prediction)
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

    recommendations = generate_recommendations(input_data)

    # Insert the prediction result and input data into MongoDB
    record = {
        'input_data': input_data,
        'prediction': predicted_class,
        'recommendations': recommendations
    }
    collection.insert_one(record)  # Save the data in MongoDB

    return jsonify({
        'prediction': predicted_class,
        'recommendations': recommendations
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
