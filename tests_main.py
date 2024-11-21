"""
tests_main.py - Test suite for the Flask application
Contains unit tests for model loading, image preprocessing, prediction functionality,
and route handling.
"""

import pytest
from app import app
from model import model, preprocess_img, predict_result
import numpy as np
from unittest.mock import patch
from PIL import Image
from io import BytesIO

@pytest.fixture
def client():
    """
    Create a test client for the Flask app.
    Returns:
        FlaskClient: A test client for making requests to the app.
    """
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_model_loading():
    """
    Verify that the ML model loads correctly.
    """
    assert model is not None

def test_preprocess_img_valid():
    """
    Test image preprocessing with valid input.
    Verifies shape and data type of processed image.
    """
    img = Image.new('RGB', (300, 300), color='red')
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    processed_img = preprocess_img(img_bytes)
    assert processed_img.shape == (1, 224, 224, 3)
    assert processed_img.dtype == np.float32

def test_preprocess_img_invalid():
    """
    Test image preprocessing with invalid input.
    Expects an OSError when non-image data is provided.
    """
    with pytest.raises(OSError):
        non_image_data = BytesIO(b"This is not an image")
        preprocess_img(non_image_data)

@patch('model.model.predict')
def test_predict_result(mock_predict):
    """
    Test prediction functionality with mocked model output.
    Verifies correct class prediction from model probabilities.
    """
    mock_predict.return_value = np.array([[0.1, 0.9]])
    predict_input = np.zeros((1, 224, 224, 3))
    result = predict_result(predict_input)
    assert result == 1

@patch('model.model.predict')
def test_predict_result_error(mock_predict):
    """
    Test prediction error handling with empty model output.
    Expects IndexError when model returns empty prediction.
    """
    mock_predict.return_value = np.array([])
    predict_input = np.zeros((1, 224, 224, 3))
    with pytest.raises(IndexError):
        predict_result(predict_input)

def test_main_route(client):
    """
    Test the main route ('/') of the application.
    Verifies response status and presence of required HTML elements.
    """
    response = client.get('/')
    assert response.status_code == 200
    assert b'Hand Sign Digit Language Detection' in response.data
    assert b'<form action="/prediction"' in response.data
    assert b'A webapp to detect a digit using hand sign language.' in response.data

@patch('model.model.predict')
def test_prediction_route(mock_predict, client):
    """
    Test the prediction route with valid image upload.
    Verifies successful prediction response and required elements.
    """
    mock_predict.return_value = [[0.1, 0.9]]
    
    img = Image.new('RGB', (300, 300), color='blue')
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    response = client.post(
        '/prediction', 
        data={'file': (img_bytes, 'test_image.jpg')},
        content_type='multipart/form-data'
    )
    assert response.status_code == 200
    assert b'Prediction' in response.data
    assert b'Hand Sign Digit Language Detection' in response.data

def test_prediction_route_method(client):
    """
    Test the prediction route with invalid HTTP method.
    Verifies proper handling of incorrect request methods.
    """
    response = client.get('/prediction')
    assert response.status_code == 405

def test_layout_elements(client):
    """
    Test presence of common layout elements.
    Verifies loading of required CSS and JavaScript files.
    """
    response = client.get('/')
    assert response.status_code == 200
    assert b'custom.css' in response.data
    assert b'bootstrap.min.js' in response.data
    assert b'jquery-3.3.1.slim.min.js' in response.data

# Acceptance Test 1: Happy Path
# Purpose: Validate the application correctly predicts the digit for a valid image file.
# Scenario: A user uploads a valid image file, and the system should return a correct prediction.
# Expected Output: HTTP 200 status and a valid prediction result (e.g., '1').
def test_acceptance_happy_path(client):
    """
    Happy path acceptance test for end-to-end prediction workflow.
    """
    with open('test_images\\1\\1.jpeg', 'rb') as img_file:
        img_bytes = BytesIO(img_file.read())

    # Send the image as a POST request
    response = client.post('/prediction', content_type='multipart/form-data',
                           data={'file': (img_bytes, '1.jpeg')})

    # Assert HTTP status code is 200
    assert response.status_code == 200
    # Assert the prediction result contains the expected digit ('1')
    assert b'1' in response.data


# Acceptance Test 2: Sad Path
# Purpose: Validate the application handles invalid input gracefully.
# Scenario: A user uploads an invalid file (non-image data), and the system should return an error message.
# Expected Output: HTTP 200 status and an error message ("File cannot be processed.").
def test_acceptance_sad_path(client):
    """
    Sad path acceptance test for handling invalid input.
    """
    invalid_file = BytesIO(b"This is not a valid image file")

    # Send the invalid file as a POST request
    response = client.post('/prediction', content_type='multipart/form-data',
                           data={'file': (invalid_file, 'invalid.txt')})

    # Assert HTTP status code is 200
    assert response.status_code == 200

    # Check for the error message in the response data (it will be in HTML)
    # The error message should appear inside the rendered HTML, possibly inside the <h2> or <div>
    assert b'File cannot be processed.' in response.data  # The error message should be in the HTML content


