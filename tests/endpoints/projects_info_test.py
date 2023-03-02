from project import create_app, db
from initialize_database_for_endpoints import create_database_for_testing
from flask_testing import TestCase


class ProjectInfoTest(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_get_info_valid_project_id(self):
        response = self.client.get('/projects/2')
        data = response.json
        expected_data = {'name': 'project',
                         'status': 'idle',
                         'train_images_amount': 0,
                         'train_annotations': 0,
                         'test_images_amount': 0,
                         'test_annotations': 1,
                         'amount_of_models': 0,
                         'total_epochs_trained': 0}

        assert response.status_code == 200
        assert data == expected_data

    def test_get_info_invalid_project_id(self):
        response = self.client.get('/projects/999')
        assert response.status_code == 400
        assert response.json == {'error': 'Project not found'}
