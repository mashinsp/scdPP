import pytest
from flask import json
from .app import app  

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'<!DOCTYPE html>' in response.data

def test_predict_valid_input(client):
    input_data = {
        'propertyType': 'Apartment',
        'city': 'City1',
        'baths': 2.5,
        'bedrooms': 3.0,
        'areaType': 'Urban',
        'areaSize': 1200.0
    }

    response = client.post('/predict', json=input_data)
    data = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200
    assert 'prediction' in data
    assert isinstance(data['prediction'], str)

def test_predict_invalid_input(client):
    input_data = {
        'propertyType': 'InvalidPropertyType',
        'city': 'City1',
        'baths': 'invalid',  # Invalid type
        'bedrooms': 3.0,
        'areaType': 'Urban',
        'areaSize': 1200.0
    }

    response = client.post('/predict', json=input_data)
    data = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200
    assert 'error' in data
    assert isinstance(data['error'], str)


if __name__ == '__main__':
    pytest.main()
