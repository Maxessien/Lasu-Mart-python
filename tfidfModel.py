import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import fetch_products, connection_wrapper
from psycopg.connection import Cursor


class TfidfModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.train()

    def train(self):
        products_list = fetch_products()
        if(not products_list or len(products_list) <= 0 or not isinstance(products_list, (list or tuple))):
            return
        products = [
            f"{product['product_name']} {product['description']} {product['category']}"
            for product in products_list
        ]
        # Fit
        self.vectorizer.fit(products)
        new_dimensions = len(self.vectorizer.vocabulary_)
        print(new_dimensions)
        
        # Clear embeddings and alter table in a single transaction
        def clear_and_alter(cur: Cursor):
            cur.execute("UPDATE products SET embedding = NULL")
            cur.execute(
                f"ALTER TABLE products ALTER COLUMN embedding TYPE VECTOR({new_dimensions})"
            )
            cur.connection.commit()
        
        try:
            connection_wrapper(clear_and_alter)
        except Exception as e:
            print(f"Error clearing embeddings and altering table: {e}")
            return
        
        # Insert embeddings with proper parameterized queries
        for product in products_list:
            encoding = self.vectorizer.transform(
                [
                    f"{product['product_name']} {product['description']} {product['category']}"
                ]
            )
            vector_list = encoding.toarray().tolist()[0]
            
            def insert_embedding(cur: Cursor):
                cur.execute(
                    "UPDATE products SET embedding = %s WHERE product_name = %s",
                    (vector_list, product['product_name'])
                )
                cur.connection.commit()
            
            connection_wrapper(insert_embedding)
        
        # Save
        joblib.dump(self.vectorizer, "tfidf_model.pkl")

    def encode(self, text):
        if not isinstance(text, list) and not isinstance(text, str):
            raise TypeError("Text must be a string or list")
        elif isinstance(text, list) and not all(
            isinstance(value, str) for value in text
        ):
            raise TypeError("Text list can only contain strings")
        
        try:
            model: TfidfVectorizer = joblib.load("tfidf_model.pkl")
            embeddings = (
                model.transform([text]) if isinstance(text, str) else model.transform(text)
            )
            return embeddings.toarray().tolist()
        except FileNotFoundError:
            print("Error: tfidf_model.pkl not found")
            return [None]
        except Exception as e:
            print(f"Error loading or using tfidf_model.pkl: {e}")
            return None
