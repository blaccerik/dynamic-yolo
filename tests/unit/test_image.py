from flask_testing import TestCase

from project.models.image import Image
from tests.initialize_test_database import create_database_for_testing
from project import db, create_app


class ImageTest(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)
        self.image = Image.query.first()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_primary_key(self):
        expected_id = 1

        actual_id = self.image.id

        assert expected_id == actual_id

    def test_image_project_id(self):
        expected_project_id = 1

        actual_project_id = self.image.project_id

        assert expected_project_id == actual_project_id

    def test_image_data(self):
        expected_image_data = bytes(10)

        actual_image_data = self.image.image

        assert expected_image_data == actual_image_data

    def test_height_data(self):
        expected_height_data = 15

        actual_height_data = self.image.height

        assert expected_height_data == actual_height_data

    def test_width_data(self):
        expected_width_data = 14

        actual_width_data = self.image.width

        assert expected_width_data == actual_width_data

    def test_subset_id(self):
        expected_subset_id = 1

        actual_subset_id = self.image.subset_id

        assert expected_subset_id == actual_subset_id
