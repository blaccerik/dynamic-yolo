from marshmallow import Schema, fields, validates_schema, ValidationError


class ProjectSettingsSchema(Schema):
    epochs = fields.Integer(required=False)
    batch_size = fields.Integer(required=False)

    train_test_ratio = fields.Float(required=False)
    pretest_size_difference_threshold = fields.Float(required=False)
    maximum_auto_train_number = fields.Integer(required=False)
    skip_pretesting = fields.Boolean(required=False)

    minimal_map_50_threshold = fields.Float(required=False)
    minimal_map_50_95_threshold = fields.Float(required=False)
    minimal_precision_threshold = fields.Float(required=False)
    minimal_recall_threshold = fields.Float(required=False)

    min_confidence_threshold = fields.Float(required=False)
    min_iou_threshold = fields.Float(required=False)

    @validates_schema
    def validate_age(self, data, **kwargs):
        if "epochs" in data and data['epochs'] <= 0:
            raise ValidationError("too small", "epoch")
        if "epochs" in data and data['epochs'] > 300:
            raise ValidationError("too big", "epoch")
        if "batch_size" in data and data['batch_size'] <= 0:
            raise ValidationError("too small", "batch_size")

        self.in_range("confidence_threshold", data)
        self.in_range("train_test_ratio", data)
        self.in_range("minimal_map_50_threshold", data)
        self.in_range("min_confidence_threshold", data)
        self.in_range("min_iou_threshold", data)
        self.in_range("pretest_size_difference_threshold", data)
        self.in_range("minimal_map_50_95_threshold", data)
        self.in_range("minimal_recall_threshold", data)
        self.in_range("minimal_precision_threshold", data)

    def in_range(self, key, data):
        if key in data:
            if data[key] < 0 or data[key] > 1:
                raise ValidationError("must be between 0 and 1", key)
