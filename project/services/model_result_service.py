from project.models.model import Model
from project.models.model_results import ModelResults
from project.models.project import Project
from project.models.subset import Subset


def retrieve_results(project_code):
    serialized_results = []
    if project_code:
        results_to_return = ModelResults.query.join(Model).filter(Model.project_id == project_code).all()

    else:
        results_to_return = ModelResults.query.all()

    for result in results_to_return:
        project_id = Project.query.get(Model.query.get(result.model_id).project_id).id
        serialized_result = {'result_id': result.id,
                             'project_id': project_id,
                             'metric_precision': result.metric_precision,
                             'metric_recall': result.metric_recall,
                             'metric_map_50': result.metric_map_50,
                             'metric_map_50_95': result.metric_map_50_95}
        serialized_results.append(serialized_result)

    return serialized_results


def retrieve_detailed_results(result_code):
    result = ModelResults.query.get(result_code)
    if not result:
        return None
    subset_name = Subset.query.get(result.subset_id).name
    project_id = Project.query.get(Model.query.get(result.model_id).project_id).id

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
                         'val_cls_loss': result.val_cls_loss
                         }
    return serialized_result
