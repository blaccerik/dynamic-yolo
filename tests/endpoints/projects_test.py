import json
from flask_testing import TestCase
from project import create_app, db
from initialize_database_for_endpoints import create_database_for_testing


class ProjectsTest(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_create_project_success(self):
        data = {
            "name": "Test Project",
            "max_class_nr": 5,
            "img_size": 640,
            "initial_model": "yolov5s"
        }
        response = self.client.post("/projects", data=json.dumps(data), content_type='application/json',
                                    follow_redirects=True)
        assert response.status_code == 201
        assert response.json == {'message': 'Project 6 created successfully'}

    def test_create_project_missing_required_fields(self):
        data = {}
        response = self.client.post("/projects", data=json.dumps(data), content_type='application/json',
                                    follow_redirects=True)
        assert response.status_code == 400

    def test_create_project_invalid_max_class_nr(self):
        data = {
            "name": "Test Project",
            "max_class_nr": -1,
            "img_size": 640,
            "initial_model": "yolov5s"
        }
        response = self.client.post("/projects", data=json.dumps(data), content_type='application/json',
                                    follow_redirects=True)
        assert response.status_code == 400

    def test_create_project_invalid_initial_model(self):
        data = {
            "name": "Test Project",
            "max_class_nr": 5,
            "img_size": 640,
            "initial_model": "invalid"
        }
        response = self.client.post("/projects", data=json.dumps(data), content_type='application/json',
                                    follow_redirects=True)
        assert response.status_code == 400

    def test_create_project_invalid_img_size(self):
        data = {
            "name": "Test Project",
            "max_class_nr": 5,
            "img_size": -1,
            "initial_model": "yolov5s"
        }
        response = self.client.post("/projects", data=json.dumps(data), content_type='application/json',
                                    follow_redirects=True)
        assert response.status_code == 400
