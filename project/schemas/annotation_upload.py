from marshmallow import Schema, fields, validates_schema, ValidationError


class AnnotationUploadSchema(Schema):
    x_center = fields.Float(required=True)
    y_center = fields.Float(required=True)
    width = fields.Float(required=True)
    height = fields.Float(required=True)
    class_id = fields.Integer(required=True)
    image_id = fields.Integer(required=True)
    uploader = fields.String(required=True)

    @validates_schema
    def validate_fields(self, data, **kwargs):

        self.in_range("x_center", data)
        self.in_range("y_center", data)
        self.in_range("width", data)
        self.in_range("height", data)

    def in_range(self, key, data):
        if key in data:
            if data[key] < 0 or data[key] > 1:
                raise ValidationError("Must be between 0 and 1", key)
