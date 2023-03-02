from project import create_app, db
from initialize_database_for_endpoints import create_database_for_testing
from flask_testing import TestCase


class ImageTest(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_get_image_default(self):
        response = self.client.get('/images/2')
        data = response.data

        assert response.status_code == 200
        assert response.content_type == 'image/png'
        assert data == b'3'

    def test_get_image_not_found(self):
        response = self.client.get('/images/1000')
        data = response.json

        assert response.status_code == 404
        assert data == {'error': 'Image not found!'}
