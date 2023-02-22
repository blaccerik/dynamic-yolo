from marshmallow import ValidationError

from project import db
from project.models.image import Image
from project.models.model import Model
from project.models.model_image import ModelImage
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
    m = Model.query.get(model_code)
    if m is None:
        raise ValidationError({"error": f"Model not found"})
    ms = ModelStatus.query.filter(ModelStatus.name.like("ready")).first()
    if m.model_status_id != ms.id:
        raise ValidationError({"error": f"Model is not in 'ready' status"})
    return m

def model_info(model_code: int):
    m = Model.query.get(model_code)
    if m is None:
        raise ValidationError({"error": f"Model not found"})
    images = db.session.query(Image.id).join(ModelImage).join(Model).filter(Model.id == model_code).all()
    image_ids = [image[0] for image in images]
    result_ids = [x.id for x in m.model_results]
    name = ModelStatus.query.get(m.model_status_id).name
    return {
        "parent_model_id": m.parent_model_id,
        "added": m.added,
        "total_epochs": m.total_epochs,
        "epochs": m.epochs,
        "model_status": name,
        "project_id": m.project_id,
        "model_results": result_ids,
        "images": image_ids
    }
