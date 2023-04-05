from flask import Blueprint, jsonify, request

from project.models.annotation import Annotation
from project.models.image import Image
from project.schemas.annotation import AnnotationSchema
from project.schemas.annotation_error_post import AnnotationErrorPost
from project.schemas.annotation_upload import AnnotationUploadSchema
from project.services.annotation_service import retrieve_annotation, change_annotation_values, \
    delete_extra_information, create_annotation, choose_between_annotations_to_keep

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


# @REQUEST_API.route('/extra/<int:annotation_id>', methods=['DELETE'])
# def remove_annotation_extras(annotation_id):
#     result = delete_extra_information(annotation_id)
#
#     if result[0] == 0:
#         return jsonify({'message': f'No errors to delete'}), 200
#
#     return jsonify({'message': f'Deleted {result[1]} model and {result[2]} human error/errors.'}), 200


@REQUEST_API.route('/<int:annotation_id>', methods=['DELETE'])
def delete_annotation(annotation_id):
    delete_extra_information(annotation_id)
    return jsonify({'message': f'Annotation deleted'}), 200


@REQUEST_API.route('/', methods=['POST'])
def add_annotation():
    data = request.json
    errors = AnnotationUploadSchema().validate(data)
    if errors:
        return jsonify({'error': f'Please check the following fields: {errors}'}), 400

    new_id = create_annotation(data)
    return jsonify({'message': f'Annotation with ID: {new_id} added'}), 200


@REQUEST_API.route('/<int:annotation_error_id>', methods=['POST'])
def check_annotation_error(annotation_error_id):
    data = request.json
    errors = AnnotationErrorPost().validate(data)
    if errors:
        return jsonify({'error': f'Please check the following fields: {errors}'}), 400

    choose_between_annotations_to_keep(annotation_error_id, data["uploader"], data["keep"])
    return jsonify({'message': f'SUCCESS'}), 200
