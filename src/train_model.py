
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {100*accuracy}%")

def save_model(model, vectorizer, model_path, vectorizer_path):
    joblib.dump(model, model_path)

    joblib.dump(vectorizer, vectorizer_path)
