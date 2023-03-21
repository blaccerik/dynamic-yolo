import io
import torch
from marshmallow import ValidationError

from project import db
from project.models.image import Image
from project.models.model import Model
from project.models.model_image import ModelImage
from project.models.model_status import ModelStatus
from project.models.project import Project
from project.models.project_status import ProjectStatus


def get_model(model_code: int):
    """
    Return the model that was asked for.
    :param model_code:
    :return:
    """
    m = Model.query.get(model_code)
    if m is None:
        raise ValidationError({"error": f"Model not found"})
    ms = ModelStatus.query.filter(ModelStatus.name.like("training")).first()
    if m.model_status_id == ms.id:
        raise ValidationError({"error": f"Model is in 'training' status"})
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


def upload_new_model(project_code, pt_file):
    project = Project.query.get(project_code)
    if not project:
        raise ValidationError('Project not found!')
    new_ps = ProjectStatus.query.filter(ProjectStatus.name.like("idle")).first()
    if project.project_status_id != new_ps.id:
        raise ValidationError("Project must be 'idle'!")

    try:
        binary_bytes = pt_file.read()
        file_in_bytes = io.BytesIO(binary_bytes)
        torch_file = torch.load(file_in_bytes)
    except:
        raise ValidationError('Corrupt file!')

    img_size_from_model = torch_file['opt']['imgsz']
    classes_from_model = torch_file['model'].names
    number_of_classes_from_model = len(torch_file['model'].names)

    project = Project.query.get(project_code)
    class_nr_from_project = project.project_settings.max_class_nr
    img_size_from_project = project.project_settings.img_size

    if img_size_from_model != img_size_from_project:
        raise ValidationError('Image size does not match!')

    if class_nr_from_project != number_of_classes_from_model:
        raise ValidationError('Class amount does not match!')

    for i in range(class_nr_from_project):
        if i not in classes_from_model:
            raise ValidationError('Classes are different!')
    model_to_upload = Model(total_epochs=0, epochs=0, model_status_id=1, project_id=project_code, model=binary_bytes)
    db.session.add(model_to_upload)
    db.session.flush()
    project.latest_model_id = model_to_upload.id
    db.session.add(project)
    db.session.commit()
