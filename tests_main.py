import os
import pytest
from flask import Flask
from app import app 
import numpy as np

# Create a test client
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# Test the prediction route with an invalid file
def test_prediction_invalid_file(client):
    response = client.post('/prediction', data={'file': 'invalid_file'})
    assert response.status_code == 200
    assert b'File cannot be processed.' in response.data

# Test to check the model loading
def test_model_loading():
    from model import model  # Import the model from model.py
    assert model is not None  # Check if the model is loaded

# Test the preprocess_img function
def test_preprocess_img():
    from model import preprocess_img
    img_path = 'test_images\\1\\1.jpeg'  # Provide a valid image path
    img = preprocess_img(img_path)
    assert img.shape == (1, 224, 224, 3)  # Check the shape of the processed image


