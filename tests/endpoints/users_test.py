import json
from flask_testing import TestCase

from project import create_app, db
from project.models.annotator import Annotator


class UsersTest(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_create_record_success(self):
        # Test creating a user with a unique name
        data = {'name': 'test_user'}
        response = self.client.post('/users/', data=json.dumps(data), content_type='application/json')
        assert response.status_code == 201
        assert response.json == {'message': 'User created successfully'}
        assert Annotator.query.filter_by(name='test_user').first() is not None

    def test_create_record_duplicate_name(self):
        # Test creating a user with a name that already exists in the database
        annotator = Annotator(name='existing_user')
        db.session.add(annotator)
        db.session.commit()

        data = {'name': 'existing_user'}
        response = self.client.post('/users/', data=json.dumps(data), content_type='application/json')
        assert response.status_code == 409
        assert response.json == {'error': 'User with that name already exists'}

    def test_create_record_invalid_data(self):
        # Test creating a user with invalid data
        data = {'email': 'test@example.com'}
        response = self.client.post('/users/', data=json.dumps(data), content_type='application/json')
        assert response.status_code == 400
        assert response.json == {'email': ['Unknown field.'], 'name': ['Missing data for required field.']}

