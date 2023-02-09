from flask_testing import TestCase

from project.models.annotator import Annotator
from tests.initialize_test_database import create_database_for_testing
from project import db, create_app


class AnnotatorTest(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)
        self.annotator = Annotator.query.filter(Annotator.name.like("human")).first()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_id(self):
        expected_id = 2

        actual_id = self.annotator.id

        assert expected_id == actual_id

    def test_name(self):
        expected_name = "human"

        actual_name = self.annotator.name

        assert expected_name == actual_name

    def test_annotations(self):
        expected_annotation_id = 2

        actual_annotation_id = self.annotator.annotations[1].id

        assert expected_annotation_id == actual_annotation_id
