from project import create_app, db
from initialize_database_for_endpoints import create_database_for_testing
from flask_testing import TestCase


class QueueTest(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_get_queue_default(self):
        response = self.client.get('/queue/')
        data = response.json
        expected_result = [{'project_id': 4, 'position': 1}, {'project_id': 5, 'position': 2}]

        assert response.status_code == 200
        assert data == expected_result

    def test_add_project_to_queue_default(self):
        response = self.client.post('/queue/', json={'project_id': 2})
        data = response.json

        assert response.status_code == 201
        assert data == {'message': 'Added project to queue'}

    def test_add_project_to_queue_project_not_found(self):
        response = self.client.post('/queue/', json={'project_id': 999})
        data = response.json

        assert response.status_code == 404
        assert data == {'error': 'Project not found!'}

    def test_add_project_to_queue_project_already_in_queue(self):
        response = self.client.post('/queue/', json={'project_id': 4})
        data = response.json

        assert response.status_code == 404
        assert data == {'error': 'Project already in queue'}
