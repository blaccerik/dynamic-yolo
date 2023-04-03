from marshmallow import Schema, fields, validates_schema, ValidationError


class ProjectSettingsSchema(Schema):
    epochs = fields.Integer(required=False)
    batch_size = fields.Integer(required=False)

    train_ratio = fields.Integer(required=False)
    val_ratio = fields.Integer(required=False)

    check_size_difference_threshold = fields.Float(required=False)
    check_center_difference_threshold = fields.Float(required=False)
    check_error_amount_threshold = fields.Float(required=False)

    maximum_auto_train_number = fields.Integer(required=False)

    min_confidence_threshold = fields.Float(required=False)
    min_iou_threshold = fields.Float(required=False)

    always_test = fields.Boolean(required=False)
    always_check = fields.Boolean(required=False)

    freeze_backbone = fields.Boolean(required=False)
    use_ram = fields.Boolean(required=False)

    devices = fields.String(required=False)

    minimal_map_50_threshold = fields.Float(required=False)
    minimal_map_50_95_threshold = fields.Float(required=False)
    minimal_precision_threshold = fields.Float(required=False)
    minimal_recall_threshold = fields.Float(required=False)

    # error weights
    model_class_confidence_weight = fields.Integer(required=False)
    human_class_confidence_weight = fields.Integer(required=False)
    annotation_confidence_weight = fields.Integer(required=False)

    image_count_weight = fields.Float(required=False)

    precision_weight = fields.Integer(required=False)
    recall_weight = fields.Integer(required=False)
    map_50_weight = fields.Integer(required=False)
    map_50_95_weight = fields.Integer(required=False)

    human_annotation_count_weight = fields.Float(required=False)

    @validates_schema
    def validate_age(self, data, **kwargs):
        self.between("epochs", data, 300)
        self.between("batch_size", data, 1000)

        self.check_ratio(data)

        self.in_range("check_size_difference_threshold", data)
        self.in_range("check_center_difference_threshold", data)
        self.in_range("check_error_amount_threshold", data)

        self.between("maximum_auto_train_number", data, 50)

        self.in_range("min_confidence_threshold", data)
        self.in_range("min_iou_threshold", data)

        self.check_bool("always_test", data)
        self.check_bool("always_check", data)
        self.check_bool("freeze_backbone", data)
        self.check_bool("use_ram", data)

        self.check_devices(data)

        self.in_range("minimal_map_50_threshold", data)
        self.in_range("minimal_map_50_95_threshold", data)
        self.in_range("minimal_recall_threshold", data)
        self.in_range("minimal_precision_threshold", data)

        self.between("model_class_confidence_weight", data, 20)
        self.between("human_class_confidence_weight", data, 20)
        self.between("annotation_confidence_weight", data, 20)

        self.in_range("image_count_weight", data)

        self.between("precision_weight", data, 20)
        self.between("recall_weight", data, 20)
        self.between("map_50_weight", data, 20)
        self.between("map_50_95_weight", data, 20)

        self.in_range("human_annotation_count_weight", data)

    def check_ratio(self, data):
        t = "train_ratio"
        v = "val_ratio"
        if t not in data and v not in data:
            return
        if t not in data:
            raise ValidationError("Missing", t)
        elif v not in data:
            raise ValidationError("Missing", v)
        a = data[t]
        b = data[v]
        if a + b >= 100:
            raise ValidationError("ratios out of range", "ratio")

    def in_range(self, key, data):
        if key in data:
            if data[key] < 0 or data[key] > 1:
                raise ValidationError("must be between 0 and 1", key)

    def between(self, key, data, e):
        if key in data:
            if data[key] <= 0 or data[key] > e:
                raise ValidationError(f"Out of range 0:{e}", key)

    def check_bool(self, key, data):
        if key in data:
            pass

    def check_devices(self, data):
        if "devices" in data:
            s = str(data["devices"])
            s_list = s.split(",")
            try:
                for i in s_list:
                    int(i)
            except:
                raise ValidationError("Correct format is '0,3,4' or '0'", "devices")

