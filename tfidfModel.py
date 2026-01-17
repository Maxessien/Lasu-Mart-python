import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import fetch_products





class TfidfModel:
    def __init__(self):
        try:
            products_list = fetch_products()
            
            #format documents
            products = [f"{product['name']} {product['description']} {product['category']}" for product in products_list]
            # Initialize TF-IDF
            vectorizer = TfidfVectorizer()

            # Fit
            vectorizer.fit(products)

            # Save
            joblib.dump(vectorizer, "tfidf_model.pkl")
            self.model = joblib.load("tfidf_model.pkl")
        except Exception as e:
            raise Exception(
                "The following error occurred: ", e)
        self.vectorizer = TfidfVectorizer()

    def retrain(self):
        products = fetch_products()
        new_matrix = self.vectorizer.fit(products)
        # Save
        joblib.dump(new_matrix, "tfidf_model.pkl")


    def encode(self, text):
        if not isinstance(text, list) and not isinstance(text, str):
            raise TypeError("Text must be a string or list")
        elif isinstance(text, list) and not all(isinstance(value, str) for value in text):
            raise TypeError("Text list can only contain strings")
        embeddings = self.model.transform([text]) if isinstance(text, str) else self.model.transform(text)

        return embeddings.toarray().tolist()
