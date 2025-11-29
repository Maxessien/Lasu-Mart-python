import pytest
from unittest import mock
from tfidfModel import TfidfModel, fetch_products
import joblib

@pytest.fixture
def mock_mongo(monkeypatch):
    class MockCollection:
        def find(self, f):
            return [{'name': 'A', 'description': 'desc', 'category': 'cat', 'price': 100}]
    class MockDB:
        def __getitem__(self, key): return MockCollection()
    class MockClient:
        def __getitem__(self, key): return MockDB()
        def close(self): pass
    monkeypatch.setattr('tfidfModel.MongoClient', lambda uri=None: MockClient())
    yield

def test_fetch_products_success(mock_mongo):
    products = fetch_products({})
    assert isinstance(products, list)
    assert len(products) > 0

def test_fetch_products_db_error(monkeypatch):
    class FailingClient:
        def __getitem__(self, key): raise Exception("fail")
        def close(self): pass
    monkeypatch.setattr('tfidfModel.MongoClient', lambda *args, **kwargs: FailingClient())
    with pytest.raises(Exception):
        fetch_products({})

def test_tfidfmodel_init(monkeypatch):
    monkeypatch.setattr('tfidfModel.fetch_products', lambda *a, **k: [{'name': 'a', 'description': 'b', 'category':'c'}])
    monkeypatch.setattr('joblib.dump', lambda model, fname: None)
    monkeypatch.setattr('joblib.load', lambda fname: mock.Mock(transform=lambda x: mock.Mock(toarray=lambda: [[0.1, 0.2]])))
    tfidf_model = TfidfModel()
    assert hasattr(tfidf_model, 'model')

def test_tfidfmodel_init_failure(monkeypatch):
    monkeypatch.setattr('tfidfModel.fetch_products', lambda *a, **k: 1/0)
    with pytest.raises(Exception):
        TfidfModel()

def test_encode_string(monkeypatch):
    monkeypatch.setattr('tfidfModel.fetch_products', lambda *a, **k: [{'name': 'test', 'description': 'd', 'category': 'cat'}])
    monkeypatch.setattr('joblib.load', lambda fname: mock.Mock(transform=lambda x: mock.Mock(toarray=lambda: [[0.1, 0.2]])))
    monkeypatch.setattr('joblib.dump', lambda model, fname: None)
    model = TfidfModel()
    result = model.encode("abc")
    assert isinstance(result, list)
    assert isinstance(result[0], list)

def test_encode_list(monkeypatch):
    monkeypatch.setattr('tfidfModel.fetch_products', lambda *a, **k: [{'name': 'test', 'description': 'd', 'category': 'cat'}])
    monkeypatch.setattr('joblib.load', lambda fname: mock.Mock(transform=lambda x: mock.Mock(toarray=lambda: [[0.1, 0.2]])))
    monkeypatch.setattr('joblib.dump', lambda model, fname: None)
    model = TfidfModel()
    result = model.encode(["abc"])
    assert isinstance(result, list)
    assert isinstance(result[0], list)

@pytest.mark.parametrize("bad_value", [None, {}, 5.5])
def test_encode_wrong_type(monkeypatch, bad_value):
    monkeypatch.setattr('tfidfModel.fetch_products', lambda *a, **k: [{'name': 'test', 'description': 'd', 'category': 'cat'}])
    monkeypatch.setattr('joblib.load', lambda fname: mock.Mock(transform=lambda x: mock.Mock(toarray=lambda: [[0.1, 0.2]])))
    monkeypatch.setattr('joblib.dump', lambda model, fname: None)
    model = TfidfModel()
    with pytest.raises(TypeError):
        model.encode(bad_value)

def test_encode_list_wrong_type(monkeypatch):
    monkeypatch.setattr('tfidfModel.fetch_products', lambda *a, **k: [{'name': 'test', 'description': 'd', 'category': 'cat'}])
    monkeypatch.setattr('joblib.load', lambda fname: mock.Mock(transform=lambda x: mock.Mock(toarray=lambda: [[0.1, 0.2]])))
    monkeypatch.setattr('joblib.dump', lambda model, fname: None)
    model = TfidfModel()
    with pytest.raises(TypeError):
        model.encode([123])

def test_retrain(monkeypatch):
    monkeypatch.setattr('tfidfModel.fetch_products', lambda *a, **k: ["abc", "def"])
    monkeypatch.setattr('joblib.dump', lambda matrix, fname: None)
    m = TfidfModel()
    m.vectorizer = mock.Mock(fit=lambda x: "dtm")
    m.retrain()

def test_retrain_integration(monkeypatch):
    monkeypatch.setattr('tfidfModel.fetch_products', lambda *a, **k: ["a", "b"])
    monkeypatch.setattr('joblib.dump', lambda matrix, fname: None)
    m = TfidfModel()
    m.vectorizer = mock.Mock(fit=lambda x: "new_matrix")
    m.retrain() # Should not throw

def test_error_output(monkeypatch):
    monkeypatch.setattr('tfidfModel.fetch_products', lambda *a, **k: 1/0)
    with pytest.raises(Exception):
        TfidfModel()

def test_model_encode_shape(monkeypatch):
    monkeypatch.setattr('tfidfModel.fetch_products', lambda *a, **k: [{'name': 'test', 'description': 'd', 'category': 'cat'}])
    monkeypatch.setattr('joblib.load', lambda fname: mock.Mock(transform=lambda x: mock.Mock(toarray=lambda: [[0.1, 0.2, 0.3]])))
    monkeypatch.setattr('joblib.dump', lambda model, fname: None)
    model = TfidfModel()
    result = model.encode("abc")
    assert len(result[0]) == 3

def test_encode_empty(monkeypatch):
    monkeypatch.setattr('tfidfModel.fetch_products', lambda *a, **k: [{'name': '', 'description': '', 'category': ''}])
    monkeypatch.setattr('joblib.load', lambda fname: mock.Mock(transform=lambda x: mock.Mock(toarray=lambda: [[0.0, 0.0]])))
    monkeypatch.setattr('joblib.dump', lambda model, fname: None)
    model = TfidfModel()
    result = model.encode("")
    assert isinstance(result, list)
    assert all(isinstance(x, list) for x in result)
