
import joblib
def load_model_and_vectorizer(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer
def predict_text(model, vectorizer, text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return prediction

if __name__ == "__main__":
    
    input_text = input("Enter the text to be checked:- \n")


    model_path = "models/trained_model.pkl"
    vectorizer_path = "models/trained_vectorizer.pkl"

    trained_model, trained_vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)

    prediction = predict_text(trained_model, trained_vectorizer, input_text)

    if prediction[0] == 0:
        print(f"The text is predicted to be human-written.")
    elif prediction[0] == 1:
        print(f"The text is predicted to be AI-generated.")
    else:
        print(f"Invalid prediction result.")
