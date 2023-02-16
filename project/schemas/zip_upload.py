from marshmallow import Schema, fields


class ZipUpload(Schema):
    split = fields.String(required=True)
    uploader_name = fields.String(required=True)