import joblib
import pymongo
import os
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer





def fetch_products(filters={}):
    try:
        uri = os.environ.get("MONGO_URI")
        client = MongoClient(uri)
        database = client["test"]
        collection = database["products"]
        cursor = collection.find(filters)
        products = list(cursor)
        # Close mongodb connection
        client.close()
        return products
    except Exception as e:
        print("An error has occured", e)
        raise Exception(e)


class TfidfModel:
    def __init__(self):
        try:
            products_list = fetch_products()
            
            #format documents
            products = [f"{product['name']} {product['description']} {product['price']} {product['category']}" for product in products_list]
            # Initialize TF-IDF
            vectorizer = TfidfVectorizer()

            # Fit
            tfidf_matrix = vectorizer.fit(products)

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


tfidf_model = TfidfModel()