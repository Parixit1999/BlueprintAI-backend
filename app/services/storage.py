"""Object storage - MinIO locally, real S3 in AWS, same code path.

The interface is small on purpose; swap implementations by config, not code.
"""
from functools import lru_cache
from typing import Protocol

from app.config import settings


class ObjectStorage(Protocol):
    def upload_bytes(self, data: bytes, key: str, content_type: str = ...) -> str: ...
    def download_bytes(self, key: str) -> bytes: ...
    def delete_bytes(self, key: str) -> None: ...
    def presigned_url(self, key: str, expires: int = ...) -> str: ...


class S3ObjectStorage:
    def __init__(self):
        import boto3

        kwargs = {"region_name": settings.aws_region}
        if settings.s3_endpoint_url:
            kwargs.update(
                endpoint_url=settings.s3_endpoint_url,
                aws_access_key_id=settings.s3_access_key,
                aws_secret_access_key=settings.s3_secret_key,
            )
        self._client = boto3.client("s3", **kwargs)
        self._bucket = settings.s3_bucket

    def upload_bytes(self, data: bytes, key: str, content_type: str = "application/octet-stream") -> str:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data, ContentType=content_type)
        return key

    def download_bytes(self, key: str) -> bytes:
        return self._client.get_object(Bucket=self._bucket, Key=key)["Body"].read()

    def delete_bytes(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def presigned_url(self, key: str, expires: int = 3600) -> str:
        return self._client.generate_presigned_url(
            "get_object", Params={"Bucket": self._bucket, "Key": key}, ExpiresIn=expires
        )


@lru_cache(maxsize=1)
def get_storage() -> ObjectStorage:
    return S3ObjectStorage()
