from marshmallow import Schema, fields


class QueueSchema(Schema):
    position = fields.Integer(required=False)
    project_id = fields.Integer(required=False)
