
from marshmallow import ValidationError
from sqlalchemy import select

from project.models.model import Model
from project.models.model_class_results import ModelClassResult
from project.models.model_results import ModelResults
from project.models.project import Project
from project.models.subset import Subset


def retrieve_results(project_code, model_code, subset_name, page_size, page_nr):
    query = ModelResults.query.join(Model).join(Subset).add_columns(Model.project_id)
    if project_code:
        p = Project.query.get(project_code)
        if p is None:
            raise ValidationError({"error": f"Project not found"})
        query = query.filter(Model.project_id == project_code)
    if model_code:
        m = Model.query.get(model_code)
        if m is None:
            raise ValidationError({"error": f"Model not found"})
        query = query.filter(Model.id == model_code)
    if subset_name:
        if subset_name not in ("val", "test"):
            raise ValidationError({"error": f"Subset can be 'train' or 'val'"})
        query = query.filter(Subset.name == subset_name)
    results_to_return = query.paginate(page=page_nr, per_page=page_size, error_out=False)
    serialized_results = []
    subset_dict = {subset.id: subset.name for subset in Subset.query.all()}
    for result, project_id in results_to_return:
        serialized_result = {'result_id': result.id,
                             "project_id": project_id,
                             'subset_name': subset_dict[result.subset_id],
                             'model_id': result.model_id,
                             'metric_precision': result.metric_precision,
                             'metric_recall': result.metric_recall,
                             'metric_map_50': result.metric_map_50,
                             'metric_map_50_95': result.metric_map_50_95}

        serialized_results.append(serialized_result)

    return serialized_results


def retrieve_detailed_results(result_code):
    result = ModelResults.query.get(result_code)
    if not result:
        raise ValidationError({'error': f'Result with the ID of {result_code} was not found!'})
    subset_name = Subset.query.get(result.subset_id).name
    project_id = Model.query.get(result.model_id).project_id

    class_results = []
    mcr = ModelClassResult.query.filter(ModelClassResult.model_results_id == result_code)
    for class_result in mcr:
        data = {
            "class_id": class_result.class_id,
            "average_precision": class_result.confidence
        }
        class_results.append(data)
    serialized_result = {'result_id': result.id,
                         'model_id': result.model_id,
                         'project_id': project_id,
                         'subset_name': subset_name,
                         'epoch': result.epoch,
                         'metric_precision': result.metric_precision,
                         'metric_recall': result.metric_recall,
                         'metric_map_50': result.metric_map_50,
                         'metric_map_50_95': result.metric_map_50_95,
                         'val_box_loss': result.val_box_loss,
                         'val_obj_loss': result.val_obj_loss,
                         'val_cls_loss': result.val_cls_loss,
                         "class_results": class_results
                         }
    return serialized_result
