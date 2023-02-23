from flask_testing import TestCase

from project.models.project_settings import ProjectSettings
from tests.initialize_test_database import create_database_for_testing
from project import db, create_app


class TestProjectSettings(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)
        self.project_settings = ProjectSettings.query.filter_by(id=1).first()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_id_column(self):
        expected_id = 1

        actual_id = self.project_settings.id

        assert expected_id == actual_id

    def test_max_class_nr_column(self):
        expected_max_class_nr = 80

        actual_max_class_nr = self.project_settings.max_class_nr

        assert expected_max_class_nr == actual_max_class_nr

    def test_epochs_column(self):
        expected_epochs = 3

        actual_epochs = self.project_settings.epochs

        assert expected_epochs == actual_epochs

    def test_batch_size_column(self):
        expected_batch_size = 8

        actual_batch_size = self.project_settings.batch_size

        assert expected_batch_size == actual_batch_size

    def test_img_size_column(self):
        expected_img_size = 640

        actual_img_size = self.project_settings.img_size

        assert expected_img_size == actual_img_size

    def test_initial_model_id_column(self):
        expected_initial_model_id = 2

        actual_initial_model_id = self.project_settings.initial_model_id

        assert expected_initial_model_id == actual_initial_model_id

    def test_confidence_threshold_column(self):
        expected_confidence_threshold = 0.95

        actual_confidence_threshold = self.project_settings.confidence_threshold

        assert expected_confidence_threshold == actual_confidence_threshold

    def test_train_test_ratio_column(self):
        expected_train_test_ratio = 0.75

        actual_train_test_ratio = self.project_settings.train_test_ratio

        assert expected_train_test_ratio == actual_train_test_ratio

    def test_minimal_map_50_threshold_column(self):
        expected_minimal_map_50_threshold = 0

        actual_minimal_map_50_threshold = self.project_settings.minimal_map_50_threshold

        assert expected_minimal_map_50_threshold == actual_minimal_map_50_threshold
        
    def test_min_confidence_threshold_column(self):
        expected_min_confidence_threshold = 0.25

        actual_min_confidence_threshold = self.project_settings.min_confidence_threshold

        assert expected_min_confidence_threshold == actual_min_confidence_threshold

    def test_min_iou_threshold_column(self):
        expected_min_iou_threshold = 0.45

        actual_min_iou_threshold = self.project_settings.min_iou_threshold

        assert expected_min_iou_threshold == actual_min_iou_threshold
