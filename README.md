---

# Credit Card Fraud Detection

### Overview
The Credit Card Fraud Detection Application is a machine learning project that uses a Random Forest Classifier to detect fraudulent credit card transactions. The application includes data preprocessing, model training, hyperparameter tuning, caching with Redis, and a web-based interface for user interaction.

### Features
- Data preprocessing with scaling and SMOTE for handling class imbalance.
- Hyperparameter tuning using Grid Search.
- Model evaluation with AUC-ROC, Precision-Recall curves.
- Redis caching to optimize repeated requests.
- Web-based interface for fraud prediction via Flask and jQuery.

### Prerequisites
- Python 3.6 or higher
- Flask web framework
- scikit-learn library
- pandas, matplotlib, seaborn for data handling and visualization
- joblib for model serialization
- jQuery for AJAX requests
- Redis for caching predictions

### Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/your-repository/credit-card-fraud-detection.git
    ```

2. Install the required Python libraries:
    ```sh
    pip install flask scikit-learn pandas matplotlib seaborn joblib redis
    ```

3. (Optional) Use Docker for easier setup:
    ```sh
    docker-compose up
    ```

### Dataset Preprocessing
1. Load and preprocess the data:
    ```python
    df = pd.read_csv('creditcard.csv')
    X = df.drop('Class', axis=1)
    y = df['Class']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    ```

2. Split the data into training and testing sets:
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

### Model Training and Evaluation
- Perform grid search for hyperparameter tuning and evaluate using classification reports and confusion matrices.
- Display performance using AUC-ROC and Precision-Recall curves.

### Web-Based Application Setup
Run the Flask app to serve model predictions through a web interface.

1. Train your model and save it:
    ```python
    joblib.dump(model, 'trained_model.pkl')
    ```

2. Start the Flask server:
    ```sh
    python app.py
    ```

3. Access the web interface via `localhost:5000`.

### Testing
Run unit tests for both the model and the API:
```sh
python -m unittest test_model.py
python -m unittest test_app.py
```

### Maintenance
- Regularly backup your trained model and database.
- Monitor for updates or improvements in the project repository.

### Contact
For any issues or suggestions, please contact the repository maintainer at [your-contact-information].

---
