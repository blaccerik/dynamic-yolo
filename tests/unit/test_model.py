from project import db, create_app
from flask_testing import TestCase
from initialize_test_database import create_database_for_testing
from project.models.model import Model


class ModelTest(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_model(self):
        model = Model.query.first()
        print(model.id)
        assert model.id == 1
