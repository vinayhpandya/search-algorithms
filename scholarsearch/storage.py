"""Local or S3 object storage; lazy boto3 import keeps offline tests lightweight."""
from __future__ import annotations
import os
from pathlib import Path
from urllib.parse import urlparse


def _s3_parts(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3" or not p.netloc:
        raise ValueError(f"Expected s3://bucket/key, got {uri!r}")
    return p.netloc, p.path.lstrip("/")


def put_bytes(uri: str, data: bytes) -> None:
    if uri.startswith("s3://"):
        import boto3
        bucket, key = _s3_parts(uri)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=data)
    else:
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
        tmp.write_bytes(data)
        tmp.replace(path)


def get_bytes(uri: str) -> bytes:
    if uri.startswith("s3://"):
        import boto3
        bucket, key = _s3_parts(uri)
        return boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    return Path(uri).read_bytes()


def exists(uri: str) -> bool:
    if uri.startswith("s3://"):
        import boto3
        from botocore.exceptions import ClientError
        bucket, key = _s3_parts(uri)
        try:
            boto3.client("s3").head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
                return False
            raise
    return Path(uri).exists()


def join(root: str, *parts: str) -> str:
    return root.rstrip("/") + "/" + "/".join(p.strip("/") for p in parts)
