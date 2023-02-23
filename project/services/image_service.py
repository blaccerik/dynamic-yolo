from project.models.image import Image


def retrieve_image(image_code: int):
    return Image.query.filter_by(id=image_code).first()
