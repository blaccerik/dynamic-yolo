from marshmallow import Schema, fields, validates_schema, ValidationError


class ProjectSettingsSchema(Schema):
    epochs = fields.Integer(required=False)
    batch_size = fields.Integer(required=False)
    img_size = fields.Integer(required=False)
    confidence_threshold = fields.Float(required=False)
    train_test_ratio = fields.Float(required=False)
    minimal_map_50_threshold = fields.Float(required=False)
    min_confidence_threshold = fields.Float(required=False)
    min_iou_threshold = fields.Float(required=False)

    @validates_schema
    def validate_age(self, data, **kwargs):
        if data['epochs'] <= 0:
            raise ValidationError("too small", "epoch")
        if data['epochs'] > 300:
            raise ValidationError("too big", "epoch")
        if data['batch_size'] <= 0:
            raise ValidationError("too small", "batch_size")
        if data['img_size'] <= 0:
            raise ValidationError("too small", "img_size")
        if data['confidence_threshold'] < 0 or data['confidence_threshold'] > 1:
            raise ValidationError("must be between 0 and 1", "confidence_threshold")
        if data['train_test_ratio'] < 0 or data['train_test_ratio'] > 1:
            raise ValidationError("must be between 0 and 1", "train_test_ratio")
        if data['minimal_map_50_threshold'] < 0 or data['minimal_map_50_threshold'] > 1:
            raise ValidationError("must be between 0 and 1", "minimal_map_50_threshold")
        if data['min_confidence_threshold'] < 0 or data['min_confidence_threshold'] > 1:
            raise ValidationError("must be between 0 and 1", "min_confidence_threshold")
        if data['min_iou_threshold'] < 0 or data['min_iou_threshold'] > 1:
            raise ValidationError("must be between 0 and 1", "min_iou_threshold ")