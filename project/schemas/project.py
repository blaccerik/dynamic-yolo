from marshmallow import Schema, fields


class Project(Schema):
    id = fields.Integer(required=False)
    name = fields.String(required=True)
    max_class_nr = fields.Integer(required=True)
