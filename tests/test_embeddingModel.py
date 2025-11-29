import pytest
from embeddingModel import encodeText

def test_encodeText_returns_none():
    # Should always return None currently
    assert encodeText("something") is None

def test_encodeText_empty_str():
    # Should always return None (API stub)
    assert encodeText("") is None

def test_encodeText_number():
    # Should always return None for non-string
    assert encodeText(123) is None

def test_encodeText_none():
    # None input returns None
    assert encodeText(None) is None


def test_encodeText_object():
    # Unhandled input returns None (API stub)
    assert encodeText({'key': 'val'}) is None
