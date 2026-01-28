from python-dotenv import load_dotenv

load_dotenv()

from  flask import Flask, request, jsonify
from flask_cors import CORS
#from FlagEmbedding import FlagAutoModel
from tfidfModel import TfidfModel
import os

# model = FlagAutoModel.from_finetuned('BAAI/bge-small-en-v1.5')


model = TfidfModel()



app = Flask(__name__)

CORS(app)


@app.route("/api/embeddings", methods=["POST"])
def create_embeddings():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "No text to encode"}), 400
    if not isinstance(data["text"], str):
        return jsonify({"error": "Text must be a string"}), 400
    embedding = model.encode(data["text"])
    return jsonify({"embedding": embedding[0]}), 201
    

@app.route("/api/new", methods=["POST"])
def add_new_prod():
    model.train()
    return jsonify({"success": True}), 200



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's port, fallback to 5000
    app.run(host="0.0.0.0", port=port, debug=True)
