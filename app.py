from flask import Flask, request, jsonify
import joblib
import numpy as np
import redis
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Connect to Redis for caching
cache = redis.StrictRedis(host='localhost', port=6379, db=0)

# Load the trained model
model = joblib.load('trained_model.pkl')

# Dummy feature scaler, replace with your real scaler
scaler = StandardScaler()

def get_cache_key(features):
    return str(features)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)

    # Check if result is in cache
    cache_key = get_cache_key(data['features'])
    cached_result = cache.get(cache_key)
    
    if cached_result:
        # Return cached result
        return jsonify({'prediction': cached_result.decode('utf-8')})
    
    # Standardize features (ensure the scaler is fitted)
    features = scaler.transform(features)
    
    # Model prediction
    prediction = model.predict(features)[0]
    
    # Cache the result
    cache.set(cache_key, str(prediction))
    
    return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
