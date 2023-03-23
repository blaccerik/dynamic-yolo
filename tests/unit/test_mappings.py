from main import SqlStream, ProjectSettings,Annotation,Prediction


def test_mappings():
    ps = ProjectSettings()
    ps.check_center_difference_threshold = 0.1
    stream = SqlStream(1,ps,1,1,1)

    annotations = [
        Prediction(Annotation(x_center=0.5, y_center=0.5), 1),
        Prediction(Annotation(x_center=0.51, y_center=0.51), 2),
        Prediction(Annotation(x_center=0.6, y_center=0.6), 4),
        Prediction(Annotation(x_center=0.5, y_center=0.6), 10),
        Prediction(Annotation(x_center=0.9, y_center=0.9), 0),
    ]
    predictions = [
        Prediction(Annotation(x_center=0.8, y_center=0.8), 0.1),
        Prediction(Annotation(x_center=0.5, y_center=0.5), 0.1),
        Prediction(Annotation(x_center=0.51, y_center=0.51), 0.3),
        Prediction(Annotation(x_center=0.59, y_center=0.61), 0.5),
        Prediction(Annotation(x_center=0.58, y_center=0.61), 0.6),
    ]

    mappings = stream.find_mappings(annotations, predictions)
    assert len(mappings) == 6

    annotations = [
        Prediction(Annotation(x_center=0.5, y_center=0.5), 1),
        Prediction(Annotation(x_center=0.51, y_center=0.51), 2),
        Prediction(Annotation(x_center=0.6, y_center=0.6), 4),
        Prediction(Annotation(x_center=0.5, y_center=0.6), 10),
        Prediction(Annotation(x_center=0.9, y_center=0.9), 0),
    ]
    predictions = [
        Prediction(Annotation(x_center=0.9, y_center=0.85), 0.1),
        Prediction(Annotation(x_center=0.5, y_center=0.5), 0.1),
    ]
    mappings = stream.find_mappings(annotations, predictions)
    assert len(mappings) == 5

    annotations = [
        Prediction(Annotation(x_center=0.5, y_center=0.5), 1),
        Prediction(Annotation(x_center=0.51, y_center=0.51), 2),
    ]
    predictions = [
        Prediction(Annotation(x_center=0.9, y_center=0.85), 0.1),
        Prediction(Annotation(x_center=0.5, y_center=0.5), 0.1),
        Prediction(Annotation(x_center=0.6, y_center=0.6), 4),
        Prediction(Annotation(x_center=0.5, y_center=0.6), 10),
        Prediction(Annotation(x_center=0.9, y_center=0.9), 0),
    ]
    mappings = stream.find_mappings(annotations, predictions)
    assert len(mappings) == 5