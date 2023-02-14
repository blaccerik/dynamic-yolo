from marshmallow import Schema, fields


class Project(Schema):
    name = fields.String(required=True)
    max_class_nr = fields.Integer(required=True)
