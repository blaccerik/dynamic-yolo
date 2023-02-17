from marshmallow import Schema, fields


class ProjectSettingsSchema(Schema):
    epochs = fields.Integer(required=False)
    batch_size = fields.Integer(required=False)
    img_size = fields.Integer(required=False)
    confidence_threshold = fields.Float(required=False)
    train_test_ratio = fields.Float(required=False)
    minimal_map_50_threshold = fields.Float(required=False)
    min_confidence_threshold = fields.Float(required=False)
    min_iou_threshold = fields.Float(required=False)
