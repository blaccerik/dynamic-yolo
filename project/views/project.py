from flask import Blueprint, render_template, request, jsonify

from project.forms import CreateProjectForm
from project.project_manager import create_project
from project.schemas.project import Project

REQUEST_API= Blueprint('project',__name__, url_prefix="/project")


@REQUEST_API.route('/', methods=['POST'])
def create_record():
    """

    """
    data = Project().load(request.json)
    name = data["name"]
    max_class_nr = data["max_class_nr"]
    msg = create_project(name, max_class_nr)
    if msg == "error":
        return jsonify({'error': 'Project with that name already exists'}), 409

    return jsonify({'message': 'Project created successfully'}), 201

# @mod.route('/create', methods=["GET", "POST"])
# def upload():
#     form = CreateProjectForm()
#     error = None
#     if form.validate_on_submit():
#         name = form.name.data
#         max_class_nr = form.max_class_nr.data
#         completed = create_project(name, max_class_nr)
#         if completed:
#             return render_template("success.html", error="done")
#         error = f"Project {name} exists"
#     return render_template('create_project.html', form=form, error=error)