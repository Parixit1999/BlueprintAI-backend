"""S3 storage for original files and evidence crops. Day 2."""


def upload_original(file_bytes: bytes, key: str) -> str:
    raise NotImplementedError


def upload_crop(image_bytes: bytes, key: str) -> str:
    raise NotImplementedError
