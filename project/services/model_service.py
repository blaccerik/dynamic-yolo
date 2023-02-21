from project.models.model import Model
from project.models.model_status import ModelStatus


def get_models():
    """
    Get all models
    :return:
    """

    models = Model().query.all()
    serialized_models = []

    # Add the model_status_name for better readability
    for model in models:
        model_dict = model.__dict__
        model_status = ModelStatus.query.get(model.model_status_id)
        model_dict['model_status_name'] = model_status.name

        result = {'id': model_dict['id'],
                  'project_id': model_dict['project_id'],
                  'model_status_name': model_dict['model_status_name'],
                  'added': model_dict['added']
                  }

        serialized_models.append(result)

    return serialized_models


def get_model(model_code: int):
    """
    Return the model that was asked for.
    :param model_code:
    :return:
    """

    return Model.query.filter_by(id=model_code).first()
