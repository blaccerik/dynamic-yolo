from marshmallow import Schema, fields


class Model(Schema):
    id = fields.Integer()
    parent_model_id = fields.Integer(allow_none=True)
    added = fields.DateTime()
    total_epochs = fields.Integer()
    epochs = fields.Integer()
    model_status_name = fields.String()
