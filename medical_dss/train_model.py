import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

#Load dataset
data = pd.read_csv('data.csv')

#separate features and target
X = data.drop('illness', axis=1)
y = data['illness']

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Evaluate accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

#save model
joblib.dump(model, 'dss_model.pkl')
print("Model saved as dss_model.pkl")