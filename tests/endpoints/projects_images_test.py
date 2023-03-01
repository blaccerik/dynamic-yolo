from project import create_app, db
from initialize_database_for_endpoints import create_database_for_testing
from flask_testing import TestCase


class ProjectImagesTest(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_get_images(self):
        response = self.client.get('/projects/3/images')
        data = response.json
        assert len(data) == 2
        assert response.status_code == 200

    def test_get_images_with_custom_page_size_and_page_number(self):
        response = self.client.get(f"/projects/3/images?page_size=10&page_nr=1")
        data = response.json

        assert len(data) == 2
        assert data[1]['id'] == 2
        assert response.status_code == 200

    def test_get_images_invalid_project_id(self):
        response = self.client.get('/projects/999/images')
        data = response.json

        assert response.status_code == 400
        assert data == {'error': 'Project not found'}

    def test_get_images_page_size_too_small(self):
        # when page size is <= 0 then return default value
        response = self.client.get(f"/projects/3/images?page_size=0&page_nr=1")
        data = response.json

        assert response.status_code == 200
        assert len(data) == 2

    def test_get_images_page_size_too_large(self):
        # when page size too large use max instead

        response = self.client.get(f"/projects/3/images?page_size=9999&page_nr=1")
        data = response.json

        assert response.status_code == 200
        assert len(data) == 2

    def test_get_image_page_number_negative(self):
        # when page number is negative set it to 1

        response = self.client.get(f"/projects/3/images?page_size=2&page_nr=-1")
        data = response.json

        assert response.status_code == 200
        assert len(data) == 2
