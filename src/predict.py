
import joblib

def load_model_and_vectorizer(model_path, vectorizer_path):
    # Load the trained model
    model = joblib.load(model_path)

    # Load the TF-IDF vectorizer
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer

def predict_text(model, vectorizer, text):
    # Vectorize the input text using the TF-IDF vectorizer
    text_tfidf = vectorizer.transform([text])

    # Make a prediction using the trained model
    prediction = model.predict(text_tfidf)

    return prediction

if __name__ == "__main__":
    
    input_text = input("Enter an essay to be checked!")


    model_path = "models/trained_model.pkl"
    vectorizer_path = "models/trained_vectorizer.pkl"

    # Load the model and vectorizer
    trained_model, trained_vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)

    # Make a prediction
    prediction = predict_text(trained_model, trained_vectorizer, input_text)

    # Display the result
    if prediction[0] == 0:
        print(f"The text is predicted to be human-written.")
    elif prediction[0] == 1:
        print(f"The text is predicted to be AI-generated.")
    else:
        print(f"Invalid prediction result.")
