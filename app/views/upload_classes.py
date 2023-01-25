from flask import render_template, request
from app import app
from app.forms import UploadClassFileForm
from app.api import upload_classes_to_db


@app.route('/upload_classes', methods=["GET", "POST"])
def upload_classes():
    form = UploadClassFileForm()
    if form.validate_on_submit():
        project_name = form.project_name.data
        file = request.files.get('file')
        content = str(file.stream.read(), 'utf-8')
        class_dict = {int(x.split()[0]): x.split()[1] for x in content.split('\n')}
        error = upload_classes_to_db(project_name, class_dict)
        return render_template("success.html", error=error)
    return render_template('upload_classes.html', form=form)