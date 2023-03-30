from project import create_app, db
from initialize_database_for_endpoints import create_database_for_testing
from flask_testing import TestCase

from project.models.project import Project
from project.models.project_settings import ProjectSettings
from project.models.project_status import ProjectStatus


class ProjectSettingsTest(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_change_project_settings(self):
        settings = {
            "epochs": 10,
            "batch_size": 16,
            "confidence_threshold": 0.5,
            "train_test_ratio": 0.2,
            "minimal_map_50_threshold": 0.5,
            "min_confidence_threshold": 0.3,
            "min_iou_threshold": 0.3
        }

        response = self.client.put(f'/projects/3/settings', json=settings)

        assert response.status_code == 201
        assert response.json == {'message': f'Successfully updated these settings: {settings}'}

    def test_change_project_settings_invalid_project_id(self):
        settings = {
            "epochs": 10,
            "batch_size": 16,
            "confidence_threshold": 0.5,
            "train_test_ratio": 0.2,
            "minimal_map_50_threshold": 0.5,
            "min_confidence_threshold": 0.3,
            "min_iou_threshold": 0.3
        }
        response = self.client.put('/projects/6/settings', json=settings)

        assert response.status_code == 400
        assert response.json == {"error": "Project not found"}

    def test_change_project_settings_invalid_data(self):
        settings = {
            "epochs": -1,
            "batch_size": 32,
            "confidence_threshold": 0.5,
            "train_test_ratio": 1.1,
            "minimal_map_50_threshold": 0.5,
            "min_confidence_threshold": 0.3,
            "min_iou_threshold": 1.1
        }
        response = self.client.put(f'/projects/3/settings', json=settings)

        assert response.status_code == 400
        assert response.json == {'error': "Please check the following fields: {'epoch': ['too small']}"}

    def test_change_project_settings_project_in_error_state(self):
        # change project to error state first
        error_ps = ProjectStatus.query.filter(ProjectStatus.name.like("error")).first()
        Project.query.filter_by(id=2).update({"project_status_id": error_ps.id})
        db.session.commit()

        settings = {
            "epochs": 10

        }
        response = self.client.put(f'/projects/2/settings', json=settings)
        idle_ps = ProjectStatus.query.filter(ProjectStatus.name.like("idle")).first()

        project = Project.query.get(2)

        assert response.status_code == 201
        assert project.project_status_id == idle_ps.id
        assert response.json == {
            "message": "Successfully updated these settings: {'epochs': 10}"}

    def test_change_project_settings_invalid_field(self):
        settings = {
            "invalid_field": 4
        }
        response = self.client.put(f'/projects/2/settings', json=settings)

        assert response.status_code == 400
        assert response.json == {'error': "Please check the following fields: {'invalid_field': ['Unknown field.']}"}

    def test_get_project_settings(self):
        response = self.client.get('/projects/2/settings')
        expected_project_settings = {
            "max_class_nr": 80,
            "epochs": 3,
            "batch_size": 8,
            "img_size": 640,
            "initial_model": 'yolov5s',
            "confidence_threshold": 0.95,
            "train_test_ratio": 0.75,
            "minimal_map_50_threshold": 0.0,
            "min_confidence_threshold": 0.25,
            "min_iou_threshold": 0.45
        }
        assert response.json == expected_project_settings

    def test_get_project_settings_invalid_project_id(self):
        response = self.client.get('/projects/999/settings')

        assert response.status_code == 400
        assert response.json == {"error": "Project not found"}