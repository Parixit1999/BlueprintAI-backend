"""Object storage - MinIO locally, real S3 in AWS, same code path.

The interface is small on purpose; swap implementations by config, not code.
"""
from functools import lru_cache
from typing import Protocol

from app.config import settings


class ObjectStorage(Protocol):
    def upload_bytes(self, data: bytes, key: str, content_type: str = ...) -> str: ...
    def upload_fileobj(self, fileobj, key: str) -> str: ...
    def download_bytes(self, key: str) -> bytes: ...
    def download_to_path(self, key: str, path: str) -> str: ...
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
        # Presigned URLs embed the endpoint host in the signature, so they must
        # be signed against the endpoint the BROWSER will hit, which differs
        # from the backend's endpoint when both run in docker.
        public = settings.s3_public_endpoint_url
        if public and public != settings.s3_endpoint_url:
            self._presign_client = boto3.client("s3", **{**kwargs, "endpoint_url": public})
        else:
            self._presign_client = self._client
        self._bucket = settings.s3_bucket

    def upload_bytes(self, data: bytes, key: str, content_type: str = "application/octet-stream") -> str:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data, ContentType=content_type)
        return key

    def upload_fileobj(self, fileobj, key: str) -> str:
        """Stream a file-like object to S3 (multipart under the hood) - a
        1 GB upload never materializes in this process's memory."""
        self._client.upload_fileobj(fileobj, self._bucket, key)
        return key

    def download_bytes(self, key: str) -> bytes:
        return self._client.get_object(Bucket=self._bucket, Key=key)["Body"].read()

    def download_to_path(self, key: str, path: str) -> str:
        """Stream an object to a local file - processing works from disk, so
        memory stays flat regardless of file size."""
        self._client.download_file(self._bucket, key, path)
        return path

    def delete_bytes(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def presigned_url(self, key: str, expires: int = 3600) -> str:
        return self._presign_client.generate_presigned_url(
            "get_object", Params={"Bucket": self._bucket, "Key": key}, ExpiresIn=expires
        )


@lru_cache(maxsize=1)
def get_storage() -> ObjectStorage:
    return S3ObjectStorage()
