from marshmallow import Schema, fields


class Upload(Schema):
    split_name = fields.String(required=True)
    uploader_name = fields.String(required=True)