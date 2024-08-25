# credit_card_fraud_detection

#### Overview
The Credit Card Fraud Detection Application is a machine learning project that uses a Random Forest Classifier to detect fraudulent credit card transactions. The application includes data preprocessing, model training, hyperparameter tuning, and a web-based interface for user interaction.

#### Prerequisites
- Python 3.6 or higher
- Flask web framework
- scikit-learn library
- pandas library
- matplotlib and seaborn libraries for data visualization
- joblib library for model serialization
- jQuery library for AJAX requests

#### Installation
1. Clone the repository to your local machine:
    ```sh
    git clone https://github.com/your-repository/credit-card-fraud-detection.git
    ```
2. Install the required Python libraries:
    ```sh
    pip install flask scikit-learn pandas matplotlib seaborn joblib
    ```
3. Load the dataset and preprocess the data:
    ```python
    df = pd.read_csv('creditcard.csv')
    X = df.drop('Class', axis=1)
    y = df['Class']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    ```
4. Split the dataset into training and testing sets:
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

#### Model Training and Evaluation
1. Initialize the Random Forest Classifier:
    ```python
    model = RandomForestClassifier()
    ```
2. Define the hyperparameters for Grid Search:
    ```python
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [4,5,6,7,8],
        'criterion' :['gini', 'entropy']
    }
    ```
3. Perform Grid Search:
    ```python
    CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    CV_rfc.fit(X_train, y_train)
    ```
4. Print the best parameters:
    ```python
    print("Best Parameters:", CV_rfc.best_params_)
    ```
5. Use the best model to make predictions:
    ```python
    model = CV_rfc.best_estimator_
    y_pred = model.predict(X_test)
    ```
6. Print the classification report and confusion matrix:
    ```python
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    ```
7. Plot the confusion matrix and feature importances:
    ```python
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]
    plt.figure(figsize=(10, 5))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    ```

#### Web-Based Application Setup
1. Create a Flask backend to serve model predictions:
    ```python
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
    ```
2. Create an HTML file with a form for user input and a JavaScript function to send data to the Flask backend:
    ```html
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Credit Card Fraud Detection</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    </head>
    <body>
        <h1>Credit Card Fraud Detection</h1>
        <form id="prediction-form">
            <!-- Add input fields for each feature -->
            <input type="text" id="feature1" placeholder="Feature 1" required>
            <input type="text" id="feature2" placeholder="Feature 2" required>
            <!-- Add more input fields as needed -->
            <button type="submit">Predict</button>
        </form>
        <div id="result"></div>

        <script>
            $(document).ready(function() {
                $('#prediction-form').submit(function(event) {
                    event.preventDefault();
                    
                    var features = [];
                    // Collect data from input fields
                    features.push(parseFloat($('#feature1').val()));
                    features.push(parseFloat($('#feature2').val()));
                    // Add more features as needed
                    
                    // Send data to Flask backend
                    $.ajax({
                        url: '/predict',
                        type: 'POST',
                        contentType: 'application/json',
                        data: JSON.stringify({'features': features}),
                        success: function(response) {
                            // Display the prediction
                            $('#result').text('Prediction: ' + response.prediction);
                        },
                        error: function(error) {
                            console.log(error);
                        }
                    });
                });
            });
        </script>
    </body>
    </html>
    ```

#### Steps to Run the Application
1. Train your model and save it using `joblib.dump(model, 'trained_model.pkl')`.
2. Run the Flask backend by executing the Python script.
3. Open the HTML file in a web browser to interact with the application.

#### Maintenance
- **Updates**: Check the GitHub repository for updates and new features.
- **Backup**: Regularly backup your trained model and database to prevent data loss.

#### Contact
For any issues or suggestions, please contact the repository maintainer at [your-contact-information].

---

This documentation provides a comprehensive guide to setting up, running, and maintaining the Credit Card Fraud Detection Application. For further customization or enhancements, refer to the source code and feel free to contribute to the project.
