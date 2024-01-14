
from src.data_preprocessing import load_data, preprocess_data, split_data, preprocess_text_data
from src.train_model import train_model, evaluate_model, save_model

def main():
    df = load_data("dataset/dataset.xlsx")
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = preprocess_text_data(X_train, X_test)

    trained_model = train_model(X_train_tfidf, y_train)

    evaluate_model(trained_model, X_test_tfidf, y_test)

    save_model(trained_model, tfidf_vectorizer, "models/trained_model.pkl", "models/trained_vectorizer.pkl")

if __name__ == "__main__":
    main()
