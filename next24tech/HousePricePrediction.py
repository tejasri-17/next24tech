# House Price Prediction using Random Forest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset (use your own or from Kaggle)
data = pd.read_csv('house_data.csv')  # Replace with your actual path

# Basic preprocessing
data = data.dropna()  # Or use Imputer
X = data.drop('Price', axis=1)
y = data['Price']

# Encode categorical features if needed
X = pd.get_dummies(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
