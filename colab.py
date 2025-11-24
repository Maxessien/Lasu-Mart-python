
import pymongo
from pymongo import MongoClient
from FlagEmbedding import FlagAutoModel
try:
    uri = "mongodb+srv://essienmax484_db_user:mygyPYVjRrfBRaGO@lasumart.jjokloj.mongodb.net/?retryWrites=true&w=majority&appName=LasuMart"
    client = MongoClient(uri)
    database = client["test"]
    collection = database["products"]

    products = collection.find({"vectorRepresentation": []})
    model = FlagAutoModel.from_finetuned('BAAI/bge-base-en-v1.5')
    track = 0
    for product in products:
        text = f"{product['name']}. {product['description']}. Price: {product['price']}. Category: {product['category']}"
        embedding = model.encode([text])
        print(f"Encoded {track+1}")
        vector = embedding[0].tolist()
        collection.update_one({"productId": product["productId"]}, {"$set": {"vectorRepresentation": vector}})
        track+=1
    client.close()
except Exception as e:
    raise Exception(
        "The following error occurred: ", e)