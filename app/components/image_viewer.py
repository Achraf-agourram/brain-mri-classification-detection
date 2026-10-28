from io import BytesIO

from PIL import Image, UnidentifiedImageError


def read_uploaded_image(uploaded_file):
    try:
        image = Image.open(BytesIO(uploaded_file.getvalue()))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            "This file could not be decoded as an image. "
            "Upload a valid PNG, JPEG, BMP, TIFF, or WebP image."
        ) from error

    return image.convert("RGB")