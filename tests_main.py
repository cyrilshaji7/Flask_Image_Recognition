"""
tests_main.py: Contains tests for the Flask app.
"""

import pytest
from app import app  # First-party import

# Create a test client
@pytest.fixture
def client():
    """Fixture to create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# Test to check the model loading
def test_model_loading():
    """Test to verify that the model is loaded properly."""
    from model import model  # Import the model from model.py
    assert model is not None  # Check if the model is loaded
