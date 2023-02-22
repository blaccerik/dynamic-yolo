from marshmallow import Schema, fields, validates_schema, ValidationError

from project.models.project_settings import ProjectSettings


class AnnotationSchema(Schema):
    x_center = fields.Float(required=False)
    y_center = fields.Float(required=False)
    width = fields.Float(required=False)
    height = fields.Float(required=False)
    class_id = fields.Integer(required=False)
    project_id = fields.Integer(required=True)

    @validates_schema
    def validate_fields(self, data, **kwargs):
        if "class_id" in data:
            project_id = data['project_id']
            project_settings = ProjectSettings.query.filter_by(id=project_id).first()
            max_class_nr = project_settings.max_class_nr
            if data["class_id"] <= 0 or data["class_id"] > max_class_nr - 1:
                raise ValidationError(
                    f"Class ID must be between 0 and {max_class_nr - 1}",
                    "class_id"
                )

        self.in_range("x_center", data)
        self.in_range("y_center", data)
        self.in_range("width", data)
        self.in_range("height", data)

    def in_range(self, key, data):
        if key in data:
            if data[key] < 0 or data[key] > 1:
                raise ValidationError("Must be between 0 and 1", key)