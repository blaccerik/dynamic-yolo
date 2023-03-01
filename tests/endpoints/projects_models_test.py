from flask_testing import TestCase
from project import create_app, db
from initialize_database_for_endpoints import create_database_for_testing


class ProjectModelsTest(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_get_all_models(self):
        response = self.client.get("/projects/3/models")
        data = response.json
        first_model_from_response = dict(list(data[0].items())[:-1])

        expected_response = {
            "id": 1,
            "project_id": 3,
            "model_status_name": 'ready'
        }

        assert response.status_code == 200
        assert first_model_from_response == expected_response
        assert len(data) == 2

    def test_get_models_invalid_project_id(self):
        # with invalid project an empty list should be returned
        response = self.client.get("/projects/999/models")
        data = response.json

        assert response.status_code == 200
        assert data == []
