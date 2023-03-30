from project import create_app, db
from initialize_database_for_endpoints import create_database_for_testing
from flask_testing import TestCase


class ProjectAnnotationsTest(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_get_annotations_default(self):
        response = self.client.get(f"/projects/3/annotations")
        data = response.json
        test_annotation_from_response = data[0]
        expected_annotation = {'annotation_id': 2,
                               'project_id': 3,
                               'annotator_name': 'human',
                               'image_id': 1
                               }
        assert response.status_code == 200
        assert len(data) == 2
        assert test_annotation_from_response == expected_annotation

    def test_get_annotations_wrong_project(self):
        # should return an empty list
        response = self.client.get(f"/projects/999/annotations")

        data = response.json
        print(data)
        assert response.status_code == 200
        assert data == []

    def test_get_annotations_invalid_page_number(self):
        response = self.client.get(f"/projects/3/annotations?page_nr=abc")

        data = response.json

        assert response.status_code == 200
        assert len(data) == 2

    def test_get_annotations_invalid_page_size(self):
        response = self.client.get(f"/projects/3/annotations?page_size=abc")

        data = response.json

        assert response.status_code == 200
        assert len(data) == 2
