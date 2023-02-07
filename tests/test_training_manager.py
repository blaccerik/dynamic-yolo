from project import create_app
from project.models.model import Model

def test_example():
    flask_app = create_app('config.TestingConfig')
    assert flask_app.config['TESTING'] is True




def test_model():
    model = Model(
        project_id=1,
        model_status_id=1,
    )
    assert model.project_id == 1
    assert model.model_status_id == 1


