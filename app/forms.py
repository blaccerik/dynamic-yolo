from flask_wtf import FlaskForm
from wtforms import MultipleFileField, StringField, SubmitField
from wtforms.validators import InputRequired


class UploadFileForm(FlaskForm):
    files = MultipleFileField("Files", validators=[InputRequired()])
    project = StringField("Project id", validators=[InputRequired()])
    uploader = StringField("Uploader", validators=[InputRequired()])
    submit = SubmitField("Upload File")
