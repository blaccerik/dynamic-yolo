from project import create_app, db
from initialize_database_for_endpoints import create_database_for_testing
from flask_testing import TestCase


class ModelsTest(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_get_model_info_default(self):
        response = self.client.get('/models/2')
        data = response.json
        data.pop('added')
        expected_result = {
            'parent_model_id': None,
            'total_epochs': 100,
            'epochs': 100,
            'model_status': 'ready',
            'project_id': 3,
            'model_results': [],
            'images': []
        }
        assert response.status_code == 200
        assert data == expected_result

    def test_get_model_info_wrong_model_id(self):
        response = self.client.get('/models/999')
        data = response.json

        assert response.status_code == 400
        assert data == {'error': 'Model not found'}

    def test_download_model_default(self):
        response = self.client.get('/models/2/download')
        actual_binary_data = response.data

        with open("tests/model_for_testing.pt", "rb") as f:
            expected_binary_data = f.read()

        assert response.status_code == 200
        assert expected_binary_data == actual_binary_data

    def test_download_model_invalid_model_id(self):
        response = self.client.get('/models/999/download')
        data = response.json

        assert response.status_code == 400
        assert data == {'error': 'Model not found'}

    def test_download_model_not_in_ready_status(self):
        response = self.client.get('/models/1/download')
        data = response.json

        assert response.status_code == 400
        assert data == {'error': "Model is not in 'ready' status"}
