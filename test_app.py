import unittest
import json
from app import app

class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict(self):
        # Sample feature input
        sample_data = {
            'features': [1.0, 0.5, 2.1, -0.8, 0.7, 1.3]  # Adjust based on model features
        }
        response = self.app.post('/predict', data=json.dumps(sample_data), content_type='application/json')
        data = json.loads(response.get_data())
        self.assertIn('prediction', data)
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
