from marshmallow import ValidationError

from project import db
from sqlalchemy import func, and_, or_
from sqlalchemy.orm import joinedload, subqueryload, aliased
from project.models.annotation import Annotation
from project.models.annotation_extra import AnnotationError
from project.models.annotator import Annotator
from project.models.image import Image
from project.models.initial_model import InitialModel
from project.models.model import Model
from project.models.model_class_results import ModelClassResult
from project.models.model_image import ModelImage
from project.models.model_results import ModelResults
from project.models.model_status import ModelStatus
from project.models.project import Project
from project.models.project_settings import ProjectSettings
from project.models.project_status import ProjectStatus
from project.models.subset import Subset


class ModelStats:

    def __init__(self, mr: ModelResults, mcr: ModelClassResult):
        self.mr = mr
        self.class_results = {mcr.class_id: mcr.confidence}

    def add(self, mcr: ModelClassResult):
        if mcr.class_id in self.class_results:
            if self.class_results[mcr.class_id] < mcr.confidence:
                self.class_results[mcr.class_id] = mcr.confidence
        else:
            self.class_results[mcr.class_id] = mcr.confidence


def create_project(name: str, class_nr: int, init_model: str, img_size: int) -> int:
    """
    Create a project
    """
    p = Project.query.filter(Project.name.like(name)).first()
    if p is not None:
        return -1

    im = InitialModel.query.filter(InitialModel.name.like(init_model)).first()
    if im is None:
        raise ValidationError(f"Unknown model {init_model}", "init_model")
    project = Project(name=name)
    db.session.add(project)
    db.session.flush()
    ps = ProjectSettings(id=project.id, max_class_nr=class_nr, initial_model_id=im.id, img_size=img_size)
    db.session.add(ps)
    db.session.commit()
    return project.id


def get_project_info(project_code: int):
    """
    Get all the important information about the project.
    :param project_code:
    :return:
    """
    project = Project.query.get(project_code)
    if project is None:
        raise ValidationError({"error": f"Project not found"})

    project_status_name = ProjectStatus.query.get(project.project_status_id).name

    test_subset_id = Subset.query.filter_by(name='test').first().id
    train_subset_id = Subset.query.filter_by(name='train').first().id
    val_subset_id = Subset.query.filter_by(name='val').first().id

    training_images = Image.query.filter_by(project_id=project_code, subset_id=train_subset_id).count()
    test_images = Image.query.filter_by(project_id=project_code, subset_id=test_subset_id).count()
    val_images = Image.query.filter_by(project_id=project_code, subset_id=val_subset_id).count()
    test_images_annotations = Annotation.query.join(Image).join(Subset).filter(and_(
        Subset.id == test_subset_id,
        Annotation.project_id == project_code
    )).count()
    training_images_annotations = Annotation.query.join(Image).join(Subset).filter(and_(
        Subset.id == train_subset_id,
        Annotation.project_id == project_code
    )).count()
    val_images_annotations = Annotation.query.join(Image).join(Subset).filter(and_(
        Subset.id == val_subset_id,
        Annotation.project_id == project_code
    )).count()
    total_models_in_project = Model.query.filter(Model.project_id == project_code).count()
    latest = project.latest_model_id
    if latest is None:
        total_epochs = 0
    else:
        total_epochs = Model.query.get(latest).total_epochs

    project_info = {
        'name': project.name,
        'status': project_status_name,
        'train_images_amount': training_images,
        'train_annotations': training_images_annotations,
        'test_images_amount': test_images,
        'test_annotations': test_images_annotations,
        'val_images_amount': val_images,
        'val_annotations': val_images_annotations,
        'amount_of_models': total_models_in_project,
        "latest_model_id": latest,
        'total_epochs_trained': total_epochs
    }
    return project_info


def get_images(project_code: int, page_size: int, page_nr: int):
    """
    Get all the images with all of its annotations and models.
    :param project_code: ID of the project
    :param page_size: Number of objects to be displayed per page
    :param page_nr: Number of the page to visit
    :return: Correctly formatted list with data
    """

    project = Project.query.get(project_code)
    if project is None:
        raise ValidationError({"error": f"Project not found"})

    data = []

    subset_dict = {}
    for subset in Subset.query.all():
        subset_dict[subset.id] = subset.name

    images_query = db.session.query(Image.id, Image.subset_id) \
        .filter(Image.project_id == project_code) \
        .limit(page_size) \
        .offset(page_size * (page_nr - 1))

    images = images_query.all()

    image_ids = [image.id for image in images]

    models_query = db.session.query(ModelImage.image_id, Model.id) \
        .join(Model) \
        .filter(ModelImage.image_id.in_(image_ids)) \
        .order_by(Model.id)

    models = models_query.all()

    annotations_query = db.session.query(Annotation.image_id, Annotation.id) \
        .filter(Annotation.image_id.in_(image_ids)) \
        .order_by(Annotation.id)

    annotations = annotations_query.all()

    for image in images:
        model_ids = [model.id for model in models if model.image_id == image.id]
        annotation_ids = [annotation.id for annotation in annotations if annotation.image_id == image.id]

        name = subset_dict[image.subset_id]
        if name is None:
            raise ValidationError({"error": f"Unknown subset ID: {image.subset_id}"})

        image_data = {
            "id": image.id,
            "annotations": annotation_ids,
            "models": model_ids,
            "subset_name": name
        }
        data.append(image_data)

    return data


def change_settings(project_code: int, new_settings: dict):
    # TODO add exceptions to get rid of this returning numbers situation
    # TODO maybe add a check in schema to filter these settings so that
    #  the values cant be 0 for example or bigger than 1
    project = Project.query.get(project_code)

    if project is None:
        raise ValidationError({"error": f"Project not found"})

    project_settings = ProjectSettings.query.get(project_code)

    if project_settings is None:
        raise ValidationError({"error": f"Project settings not found"})

    for k, v in new_settings.items():
        setattr(project_settings, k, v)

    # update project if its in error state
    idle_ps = ProjectStatus.query.filter(ProjectStatus.name.like("idle")).first()
    error_ps = ProjectStatus.query.filter(ProjectStatus.name.like("error")).first()
    if project.project_status_id == error_ps.id:
        project.project_status_id = idle_ps.id
        db.session.add(project)
    db.session.add(project_settings)
    db.session.commit()


def get_settings(project_code: int) -> dict:
    project = Project.query.get(project_code)

    if project is None:
        raise ValidationError({"error": f"Project not found"})
    project_settings = ProjectSettings.query.get(project_code)

    if project_settings is None:
        raise ValidationError({"error": f"Project settings not found"})

    name = InitialModel.query.get(project_settings.initial_model_id).name

    return {
        "max_class_nr": project_settings.max_class_nr,
        "epochs": project_settings.epochs,
        "batch_size": project_settings.batch_size,
        "img_size": project_settings.img_size,
        "initial_model": name,

        "train_ratio": project_settings.train_ratio,
        "val_ratio": project_settings.val_ratio,

        "check_size_difference_threshold": project_settings.check_size_difference_threshold,
        "check_center_difference_threshold": project_settings.check_center_difference_threshold,
        "check_error_amount_threshold": project_settings.check_error_amount_threshold,

        "maximum_auto_train_number": project_settings.maximum_auto_train_number,

        "min_confidence_threshold": project_settings.min_confidence_threshold,
        "min_iou_threshold": project_settings.min_iou_threshold,

        "always_test": project_settings.always_test,
        "always_check": project_settings.always_check,

        "freeze_backbone": project_settings.freeze_backbone,
        "use_ram": project_settings.use_ram,

        "devices": project_settings.devices,

        "minimal_map_50_threshold": project_settings.minimal_map_50_threshold,
        "minimal_map_50_95_threshold": project_settings.minimal_map_50_95_threshold,
        "minimal_precision_threshold": project_settings.minimal_precision_threshold,
        "minimal_recall_threshold": project_settings.minimal_recall_threshold,

        "model_class_confidence_weight": project_settings.model_class_confidence_weight,
        "human_class_confidence_weight": project_settings.human_class_confidence_weight,
        "annotation_confidence_weight": project_settings.annotation_confidence_weight,

        "image_count_weight": project_settings.image_count_weight,

        "precision_weight": project_settings.precision_weight,
        "recall_weight": project_settings.recall_weight,
        "map_50_weight": project_settings.map_50_weight,
        "map_50_95_weight": project_settings.map_50_95_weight,

        "human_annotation_count_weight": project_settings.human_annotation_count_weight
    }


def get_all_projects():
    """
    Get all projects
    """
    return Project.query.all()


def get_models(project_code, page_nr, page_size):
    """
    Get all models
    :return:
    """

    models = Model().query.filter(Model.project_id == project_code).paginate(page=page_nr, per_page=page_size,
                                                                             error_out=False)
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


def retrieve_annotations(project_code, page_nr, page_size):
    """
    Get all annotations.
    :return:
    """
    annotations_and_annotators = db.session.query(Annotation, Annotator) \
        .join(Annotator, Annotation.annotator_id == Annotator.id) \
        .filter(
        Annotation.project_id == project_code) \
        .paginate(page=page_nr, per_page=page_size, error_out=False)

    annotations_to_return = []

    for annotation, annotator in annotations_and_annotators:
        annotation_id = annotation.id
        project_id = annotation.project_id
        annotator_name = annotator.name
        image_id = annotation.image_id
        annotations_to_return.append({"annotation_id": annotation_id,
                                      "project_id": project_id,
                                      "annotator_name": annotator_name,
                                      "image_id": image_id})

    return annotations_to_return


def find_score(project_settings: ProjectSettings,
               model_class_confidence,
               human_class_confidence,
               confidence,
               image_count,
               mr: ModelResults,
               human_ano_count):

    score = model_class_confidence * project_settings.model_class_confidence_weight + \
            human_class_confidence * project_settings.human_class_confidence_weight + \
            confidence * project_settings.annotation_confidence_weight + \
            image_count * project_settings.image_count_weight + \
            mr.metric_precision * project_settings.precision_weight + \
            mr.metric_recall * project_settings.recall_weight + \
            mr.metric_map_50 * project_settings.map_50_weight + \
            mr.metric_map_50_95 * project_settings.map_50_95_weight + \
            human_ano_count * project_settings.human_annotation_count_weight
    return score


def retrieve_project_errors(project_code, page_nr, page_size):
    project = Project.query.get(project_code)

    if project is None:
        raise ValidationError({"error": f"Project not found"})

    project_settings = ProjectSettings.query.get(project_code)
    if project_settings is None:
        raise ValidationError({"error": f"Project settings not found"})

    # Create aliases for the Annotation table
    a1 = aliased(Annotation)
    a2 = aliased(Annotation)

    # Perform the left join query
    query = db.session.query(AnnotationError, a1, a2). \
        join(a1, AnnotationError.model_annotation_id == a1.id, isouter=True). \
        join(a2, AnnotationError.human_annotation_id == a2.id, isouter=True). \
        filter(or_(a1.project_id == project_code, a2.project_id == project_code)).all()

    # get all model results
    ss_val = Subset.query.filter(Subset.name.like("val")).first()
    results = db.session.query(ModelClassResult, ModelResults). \
        join(ModelResults, ModelResults.id == ModelClassResult.model_results_id). \
        filter(ModelResults.subset_id == ss_val.id).all()
    model_results = {}
    for mcr, mr in results:
        model_id = mr.model_id
        if model_id in model_results:
            model_results[model_id].add(mcr)
        else:
            model_results[model_id] = ModelStats(mr, mcr)

    weighted_results = []
    # go through every mistake and give it a score
    for ae, am, ah in query:
        model_id = ae.model_id
        # model doesnt have class results
        if model_id not in model_results:
            continue

        model_stats = model_results[model_id]
        confidence = ae.confidence

        if confidence is None:
            confidence = 1

        human_annotation_count = ae.human_annotation_count

        if human_annotation_count is None:
            human_annotation_count = 0

        image_count = ae.image_count

        model_class_confidence = 0
        if am is not None:
            model_class_confidence = model_stats.class_results[am.class_id]


        human_class_confidence = 1
        if ah is not None:
            human_class_confidence = model_stats.class_results[ah.class_id]
        score = find_score(project_settings, model_class_confidence, human_class_confidence, confidence, image_count, model_stats.mr, human_annotation_count)
        weighted_results.append((score, ae))

    weighted_results.sort(key=lambda x: x[0], reverse=True)

    start_index = (page_nr - 1) * page_size
    end_index = start_index + page_size
    return weighted_results[start_index:end_index]
