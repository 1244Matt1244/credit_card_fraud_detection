from flask import Flask, request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict(data['features'])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
