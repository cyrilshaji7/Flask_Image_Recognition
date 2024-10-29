"""
tests_main.py: Contains tests for the Flask app.
"""

import pytest
from app import app

# Create a test client
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as test_client:
        yield test_client

# Test the prediction route with an invalid file
def test_prediction_invalid_file(client):
    response = client.post('/prediction', data={'file': 'invalid_file'})
    assert response.status_code == 200
    assert b'File cannot be processed.' in response.data

# Test to check the model loading
def test_model_loading():
    from model import model  # Import the model from model.py
    assert model is not None  # Check if the model is loaded
