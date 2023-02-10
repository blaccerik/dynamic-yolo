from flask_testing import TestCase

from project.models.model_image import ModelImage
from tests.initialize_test_database import create_database_for_testing
from project import db, create_app


class TestModelImage(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)
        self.model_image = ModelImage.query.first()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_model_id_column(self):
        expected_model_id = 1

        actual_model_id = self.model_image.model_id

        assert expected_model_id == actual_model_id

    def test_image_id_column(self):
        expected_image_id = 1

        actual_image_id = self.model_image.image_id

        assert expected_image_id == actual_image_id

    def test_image_subset_id_column(self):
        expected_image_subset_id = 1

        actual_image_subset_id = self.model_image.image_subset_id

        assert expected_image_subset_id == actual_image_subset_id
