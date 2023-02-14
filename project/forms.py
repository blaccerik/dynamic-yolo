from flask_wtf import FlaskForm
from wtforms import MultipleFileField, StringField, SubmitField, FileField, IntegerField
from wtforms.validators import InputRequired


class UploadFileForm(FlaskForm):
    files = MultipleFileField("Files", validators=[InputRequired()])
    project = StringField("Project id", validators=[InputRequired()])
    uploader = StringField("Uploader", validators=[InputRequired()])
    split = StringField("Split", validators=[InputRequired()])
    submit = SubmitField("Upload File")


class UploadClassFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    project_name = StringField("Project name", validators=[InputRequired()])
    submit = SubmitField("Upload File")

class CreateUserForm(FlaskForm):
    name = StringField("Name", validators=[InputRequired()])
    submit = SubmitField("Create")

class CreateProjectForm(FlaskForm):
    name = StringField("Name", validators=[InputRequired()])
    max_class_nr = IntegerField("classes", validators=[InputRequired()])
    submit = SubmitField("Create")
