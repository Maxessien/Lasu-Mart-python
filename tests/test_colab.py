import pytest
from unittest.mock import MagicMock, patch
import colab

@patch('colab.MongoClient')
@patch('colab.FlagAutoModel')
def test_embedding_pipeline_success(FlagAutoModelMock, MongoClientMock):
    model_instance = MagicMock()
    model_instance.encode.return_value = [[0.1, 0.2, 0.3]]
    FlagAutoModelMock.from_finetuned.return_value = model_instance

    mock_collection = MagicMock()
    mock_collection.find.return_value = [{'name': 'x', 'description': 'desc', 'price': 10, 'category': 'prod', 'productId': 42, 'vectorRepresentation': []}]
    mock_collection.update_one.return_value = None
    mock_db = {'products': mock_collection}
    MongoClientMock.return_value = {'test': mock_db}
    
    try:
        exec(open('colab.py').read())
    except Exception:
        pytest.fail("Should not raise exception")

@patch('colab.MongoClient')
@patch('colab.FlagAutoModel')
def test_embedding_pipeline_with_empty_products(FlagAutoModelMock, MongoClientMock):
    model_instance = MagicMock()
    model_instance.encode.return_value = [[0.0, 0.0, 0.0]]
    FlagAutoModelMock.from_finetuned.return_value = model_instance

    mock_collection = MagicMock()
    mock_collection.find.return_value = []
    mock_collection.update_one.return_value = None
    mock_db = {'products': mock_collection}
    MongoClientMock.return_value = {'test': mock_db}

    exec(open('colab.py').read())

@patch('colab.MongoClient')
@patch('colab.FlagAutoModel')
def test_pipeline_db_error(FlagAutoModelMock, MongoClientMock):
    FlagAutoModelMock.from_finetuned.side_effect = Exception("model error")
    MongoClientMock.return_value = {'test': {}}
    with pytest.raises(Exception):
        exec(open('colab.py').read())


def test_pipeline_mongo_fail(monkeypatch):
    monkeypatch.setattr('colab.MongoClient', lambda *a, **k: (_ for _ in ()).throw(Exception("Mongo fail")))
    with pytest.raises(Exception):
        exec(open('colab.py').read())


def test_pipeline_bad_product(monkeypatch):
    def fake_coll(*a, **k):
        class C:
            def find(self, _):
                return [{'name': None, 'description': None, 'price': None, 'category': None, 'productId': 1, 'vectorRepresentation': []}]
            def update_one(self, *a, **k): return None
        class D:
            def __getitem__(self, key): return C()
        class Client:
            def __getitem__(self, key): return D()
        def close(self): pass
    monkeypatch.setattr('colab.MongoClient', fake_coll)
    with pytest.raises(Exception):
        exec(open('colab.py').read())
