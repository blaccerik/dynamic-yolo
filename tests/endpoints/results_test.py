from project import create_app, db
from initialize_database_for_endpoints import create_database_for_testing
from flask_testing import TestCase


class ResultsTest(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_get_results_with_no_project_id(self):
        response = self.client.get('/results/')
        data = response.json

        assert response.status_code == 200
        assert len(data) == 2

    def test_get_results_with_project_id(self):
        response = self.client.get('/results/?project_id=2')
        data = response.json

        assert response.status_code == 200
        assert data[0]['project_id'] == 2

    def test_get_results_with_invalid_project_id(self):
        response = self.client.get('/results/?project_id=999')
        data = response.json

        assert response.status_code == 200
        assert data == []

    def test_get_detailed_results_default(self):
        response = self.client.get('/results/2')
        data = response.json

        expected_result = \
            {'result_id': 2,
             'model_id': 3,
             'project_id': 2,
             'subset_name': 'test',
             'epoch': None,
             'metric_precision': 0.2,
             'metric_recall': 0.3,
             'metric_map_50': 0.4,
             'metric_map_50_95': 0.5,
             'val_box_loss': 0.6,
             'val_obj_loss': 0.7,
             'val_cls_loss': 0.8}

        assert response.status_code == 200
        assert data == expected_result

    def test_get_detailed_results_invalid_id(self):
        response = self.client.get('/results/999')
        data = response.json

        assert response.status_code == 404
        assert data == {'error': f'Result with the ID of 999 was not found!'}
