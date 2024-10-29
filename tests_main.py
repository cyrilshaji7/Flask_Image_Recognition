import os
import pytest
import numpy as np
from app import app  # First-party import

# Create a test client
@pytest.fixture
def client():
    """Fixture to create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Test the prediction route with an invalid file
def test_prediction_invalid_file(client):
    """Test that the prediction route returns an error for an invalid file."""
    response = client.post('/prediction', data={'file': 'invalid_file'})
    assert response.status_code == 200
    assert b'File cannot be processed.' in response.data

# Test to check the model loading
def test_model_loading():
    """Test to verify that the model is loaded properly."""
    from model import model  # Import the model from model.py
    assert model is not None  # Check if the model is loaded
