from marshmallow import Schema, fields


class Project(Schema):
    id = fields.Integer(required=True)
    name = fields.String(required=True)
    max_class_nr = fields.Integer(required=True)
