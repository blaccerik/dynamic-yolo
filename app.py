from flask import send_from_directory, make_response, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
from marshmallow import ValidationError

from project import create_app
app = create_app('config.DevelopmentConfig')

@app.route("/static/<path:path>")
def send_static(path):
    """
    Swagger path
    """
    return send_from_directory("static", path)

@app.errorhandler(ValidationError)
def handle_validation_error(error):
    """
    Return serialization error
    """
    return make_response(jsonify(error.messages), 400)

### swagger specific ###
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Dynamic yolo swagger"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)
### end swagger specific ###
