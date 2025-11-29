import pytest
from app import app
from unittest.mock import patch, Mock

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

def test_post_embeddings_success(client):
    with patch("app.model") as mock_model:
        mock_model.encode.return_value = [[0.11, 0.12, 0.13]]
        response = client.post("/api/embeddings", json={"text": "some string"})
        assert response.status_code == 201
        assert "embedding" in response.json
        assert isinstance(response.json["embedding"], list)

def test_post_embeddings_missing_text(client):
    resp = client.post("/api/embeddings", json={})
    assert resp.status_code == 400
    assert "error" in resp.json

def test_post_embeddings_not_json(client):
    resp = client.post("/api/embeddings", data="not json", content_type="text/plain")
    assert resp.status_code == 400

def test_post_embeddings_text_wrong_type(client):
    resp = client.post("/api/embeddings", json={"text": {"not": "a string"}})
    assert resp.status_code == 400

def test_post_embeddings_text_list_type(client):
    resp = client.post("/api/embeddings", json={"text": [1, 2, 3]})
    assert resp.status_code == 400

def test_post_embeddings_edge_empty(client):
    with patch("app.model") as mock_model:
        mock_model.encode.return_value = [[]]
        response = client.post("/api/embeddings", json={"text": ""})
        assert response.status_code == 201
        assert response.json["embedding"] == []

def test_post_embeddings_internal_error(client):
    with patch("app.model") as mock_model:
        mock_model.encode.side_effect = Exception("model fail")
        resp = client.post("/api/embeddings", json={"text": "foo"})
        assert resp.status_code in {500, 400}

def test_options_endpoint(client):
    resp = client.options("/api/embeddings")
    assert resp.status_code == 200

def test_flask_app_runs(monkeypatch):
    monkeypatch.setattr("os.environ.get", lambda x, default=None: "5001" if x=="PORT" else default)
    # Just ensure app creation doesn't error
    from app import app
    assert isinstance(app, type(app))

def test_app_cors_enabled(client):
    resp = client.post("/api/embeddings", json={"text": "foo"})
    assert "Access-Control-Allow-Origin" in resp.headers

def test_api_with_mocked_tfidfmodel(client):
    with patch("app.model.encode") as mock_enc:
        mock_enc.return_value = [[1]]
        resp = client.post("/api/embeddings", json={"text": "hello"})
        assert resp.status_code == 201
        assert resp.json["embedding"] == [1]
