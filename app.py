from  flask import Flask, request, jsonify
from flask-cors import CORS
from emmbeddingModel import encodeText



app = Flask(__name__)

CORS(app)


@app.route("/api/embeddings", methods=["POST"])
def create_embeddings():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "No text to encode"}), 401
    if not isinstance(data["text"], str):
        return jsonify({"error": "Text must be a string"}), 401
    embedding = encodeText(data["text"])
    return jsonify({"embedding": embedding}), 201
    
    
    
    



if __name__ == "__main__":
    app.run(debug=True)