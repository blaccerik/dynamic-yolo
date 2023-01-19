from io import BytesIO
from time import sleep

import PIL.Image
from flask import render_template, request
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, MultipleFileField, StringField
from wtforms.validators import InputRequired

from app import app
from app.api import upload_files
from app.forms import UploadFileForm
from app.models.annotation import Annotation
from app.models.image import Image



image_types = [
    "image/jpeg",
    "image/png"
]

text_types = [
    "text/plain"
]


def _filter_files(files):
    for f in files:
        if f.content_type in image_types or f.content_type in text_types:
            for item in _filestorage_to_db_item(f):
                yield item


def _text_to_annotations(text, name):
    _list = []
    for line in text.splitlines():
        try:
            nr, x, y, w, h = line.strip().split(" ")
            nr = int(nr)
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            _list.append((Annotation(x_center=x, y_center=y, width=w, height=h, class_id=nr), name))
        except Exception as e:
            continue
    return _list


def _filestorage_to_db_item(f):
    content = f.stream.read()
    if f.content_type in text_types:  # text file
        text = str(content, "utf-8")
        name = secure_filename(f.filename).split(".")[0]
        _list = _text_to_annotations(text, name)
        f.stream.close()
        return _list
    else:  # image file
        io_bytes = BytesIO(content)
        img = PIL.Image.open(io_bytes)
        f.stream.close()
        name = secure_filename(f.filename).split(".")[0]
        return (Image(image=content, width=img.size[0], height=img.size[1]), name),


@app.route('/upload', methods=["GET", "POST"])
def upload():
    form = UploadFileForm()
    if form.validate_on_submit():
        project_name = form.project.data
        uploader = form.uploader.data
        uploaded_files = request.files.getlist("files")
        uploaded_files = [f for f in _filter_files(uploaded_files)]
        error = upload_files(uploaded_files, project_name, uploader)
        return render_template("success.html", error=error)
    return render_template('upload.html', form=form)
