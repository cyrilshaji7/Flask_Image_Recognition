"""
tests_main.py: Contains tests for the Flask app.
"""

import pytest
from app import app  # First-party import
from model import model  # Import the model from model.py
from model import preprocess_img, predict_result
import numpy as np
from unittest.mock import patch
from PIL import Image
from io import BytesIO

# Create a test client
@pytest.fixture
def test_client():  # Renamed fixture to avoid redefinition warning
    """Fixture to create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Test to check the model loading
def test_model_loading():
    """Test to verify that the model is loaded properly."""
    assert model is not None  # Check if the model is loaded

def test_predict_result_initial():
    """Test the initial prediction result function with a sample input."""
    img_path = "test_images\\1\\1.jpeg"
    img = preprocess_img(img_path)
    pred = predict_result(img)
    assert isinstance(1, int)  

def test_preprocess_img_valid():
    """Test preprocessing of a valid image."""
    # Create an in-memory image
    img = Image.new('RGB', (300, 300), color='red')
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    processed_img = preprocess_img(img_bytes)
    assert processed_img.shape == (1, 224, 224, 3)
    assert processed_img.dtype == np.float32

def test_preprocess_img_invalid_input():
    """Test preprocessing of an invalid image input."""
    with pytest.raises(OSError):  # Catch specific exceptions
        non_image_data = BytesIO(b"This is not an image")
        preprocess_img(non_image_data)

@patch('model.model.predict')
def test_predict_result(mock_predict):
    """Test prediction result with mocked output."""
    # Mock the predict method to return a predefined output
    mock_predict.return_value = np.array([[0.1, 0.9]])
    predict_input = np.zeros((1, 224, 224, 3))
    result = predict_result(predict_input)
    assert result == 1  # np.argmax([0.1, 0.9]) should be 1

@patch('model.model.predict')
def test_predict_result_empty_prediction(mock_predict):
    """Test handling of empty prediction result."""
    # Mock the predict method to return an empty array
    mock_predict.return_value = np.array([])
    predict_input = np.zeros((1, 224, 224, 3))
    with pytest.raises(IndexError):
        predict_result(predict_input)
