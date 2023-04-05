from marshmallow import Schema, fields, validates_schema, ValidationError

class AnnotationErrorPost(Schema):
    uploader = fields.String(required=True)
    keep = fields.String(required=True)

    @validates_schema
    def validate_fields(self, data, **kwargs):
        possible_keep_values = ['model', 'human', 'both', "none"]
        if data["keep"] not in possible_keep_values:
            raise ValidationError(f'Possible values are {possible_keep_values}', "keep")