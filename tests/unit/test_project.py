from flask_testing import TestCase

from project.models.project import Project
from tests.initialize_test_database import create_database_for_testing
from project import db, create_app


class TestProject(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)
        self.project = Project.query.filter_by(name="unknown").first()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_id_column(self):
        expected_id = 1

        actual_id = self.project.id

        assert expected_id == actual_id

    def test_name_column(self):
        expected_name = "unknown"

        actual_name = self.project.name

        assert expected_name == actual_name

    def test_latest_model_id_column(self):
        expected_latest_model_id = 1

        actual_latest_model_id = self.project.latest_model_id

        assert expected_latest_model_id == actual_latest_model_id

    def test_queue_relationship(self):
        expected_queue_position = 1

        actual_queue_position = self.project.queue.position

        assert expected_queue_position == actual_queue_position

    def test_project_settings_relationship(self):
        expected_project_settings_id = 1

        actual_project_settings_id = self.project.project_settings.id

        assert expected_project_settings_id == actual_project_settings_id

    def test_image_classes_relationship(self):
        expected_image_class_length = 2

        actual_image_class_length = len(self.project.image_classes)

        assert expected_image_class_length == actual_image_class_length

    def test_images_relationship(self):
        expected_image_id = 1

        actual_image_id = self.project.images[0].id

        assert expected_image_id == actual_image_id

    def test_annotations_relationship(self):
        expected_annotations_length = 2

        actual_annotations_length = len(self.project.annotations)

        assert expected_annotations_length == actual_annotations_length

    def test_models_relationship(self):
        expected_models_length = 2

        actual_models_length = len(self.project.models)

        assert expected_models_length == actual_models_length

    def test_latest_model_relationship(self):
        expected_latest_model_id = 1

        actual_latest_model_id = self.project.latest_model.id

        assert expected_latest_model_id == actual_latest_model_id

    def test_default_project_status_id(self):
        expected_project_status_id = 2

        actual_project_status_id = self.project.project_status_id

        assert expected_project_status_id == actual_project_status_id
