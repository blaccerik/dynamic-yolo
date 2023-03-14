from flask import Blueprint, jsonify, request

from project.models.annotation import Annotation
from project.schemas.annotation import AnnotationSchema
from project.services.annotation_service import retrieve_annotation, change_annotation_values

REQUEST_API = Blueprint('annotations', __name__, url_prefix="/annotations")


@REQUEST_API.route('/<int:annotation_id>', methods=['GET'])
def get_annotation(annotation_id):
    annotation = retrieve_annotation(annotation_id)
    if not annotation:
        return jsonify({'error': 'Check the annotation ID.'}), 404

    return annotation


@REQUEST_API.route('/<int:annotation_id>', methods=['PUT'])
def change_annotation(annotation_id):
    data = request.json
    annotation = Annotation.query.get(annotation_id)
    if not annotation:
        return jsonify({'error': f'Annotation not found!'}), 404

    project_id = annotation.project_id
    data['project_id'] = project_id
    errors = AnnotationSchema().validate(data)
    if errors:
        return jsonify({'error': f'Please check the following fields: {errors}'}), 400

    data.pop('project_id')
    change_annotation_values(annotation_id, data)
    return jsonify({'message': f'Successfully updated these settings: {data}'}), 200
