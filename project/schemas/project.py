from marshmallow import Schema, fields, validates_schema, ValidationError


class Project(Schema):
    id = fields.Integer(required=False)
    name = fields.String(required=True)
    max_class_nr = fields.Integer(required=True)
    initial_model = fields.String(required=True)
    img_size = fields.Integer(required=True)

    @validates_schema
    def validate_age(self, data, **kwargs):

        if data["max_class_nr"] <= 0:
            raise ValidationError("too small", "max_class_nr")
        _list = ["yolov5n", "yolov5s","yolov5m","yolov5l","yolov5x"]
        if data["initial_model"] not in _list:
            raise ValidationError(f"Allowed values are: {_list}", "initial_model")
        if data['img_size'] <= 0:
            raise ValidationError("too small", "img_size")
