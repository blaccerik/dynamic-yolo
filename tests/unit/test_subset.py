from flask_testing import TestCase

from project.models.subset import Subset
from tests.initialize_test_database import create_database_for_testing
from project import db, create_app


class TestSubset(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)
        self.subset = Subset.query.filter_by(name="train").first()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_name_column(self):
        expected_name = "train"

        actual_name = self.subset.name

        assert expected_name == actual_name

    def test_id_column(self):
        expected_id = 1

        actual_id = self.subset.id

        assert expected_id == actual_id

    def test_model_images_column(self):
        expected_model_image_id = 1

        actual_model_image_id = self.subset.model_images[0].id

        assert expected_model_image_id == actual_model_image_id

    def test_model_results_relationship(self):
        expected_model_result_id = 2

        actual_model_result_id = self.subset.model_results[1].id

        assert expected_model_result_id == actual_model_result_id
