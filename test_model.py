import unittest
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class TestModel(unittest.TestCase):
    def setUp(self):
        # Load data and model
        self.model = joblib.load('trained_model.pkl')
        df = pd.read_csv('creditcard.csv')
        X = df.drop('Class', axis=1)
        y = df['Class']
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    def test_prediction_shape(self):
        prediction = self.model.predict(self.X_test)
        self.assertEqual(prediction.shape, self.y_test.shape, "Prediction shape mismatch")
    
    def test_accuracy(self):
        accuracy = self.model.score(self.X_test, self.y_test)
        self.assertGreaterEqual(accuracy, 0.8, "Accuracy is lower than 80%")

if __name__ == '__main__':
    unittest.main()
