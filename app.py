from  flask import Flask, request, jsonify
from flask-cors import CORS
from emmbeddingModel import encodeText



app = Flask(__name__)

CORS(True)


@app.route("/api/embeddings", methods=["POST"])
def create_embeddings():
    text = request.json
    embedding = encodeText(text["text"])
    return jsonify({"embedding": embedding}), 201
    
    
    
    



if __name__ == "__main__":
    app.run(debug=True)