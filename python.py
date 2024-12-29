import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    dataset_path = r"C:\Users\Manian VJS\Desktop\Datasets\final (1).csv"
    sports_data = pd.read_csv(dataset_path)
    
    X = pd.get_dummies(sports_data[['runs', 'opponent', 'ground', 'date', 'match', 'Match_No']], columns=['match'], drop_first=True)
    y = sports_data['total']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    return jsonify({'Mean Squared Error': mse})

if __name__ == '__main__':
    app.run(debug=True)
