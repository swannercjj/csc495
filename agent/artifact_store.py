import os
import shutil
import tempfile
from urllib.parse import urlparse

import torch

try:
    import boto3  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency for cloud runs
    boto3 = None


def is_s3_uri(path):
    return isinstance(path, str) and path.startswith("s3://")


def split_s3_uri(uri):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def join_artifact_path(base_path, *parts):
    clean_parts = [str(part).strip("/") for part in parts if part not in (None, "")]
    if is_s3_uri(base_path):
        bucket, key_prefix = split_s3_uri(base_path)
        key_parts = [key_prefix] if key_prefix else []
        key_parts.extend(clean_parts)
        key = "/".join(part for part in key_parts if part)
        return f"s3://{bucket}/{key}" if key else f"s3://{bucket}"
    return os.path.join(base_path, *clean_parts)


def ensure_parent_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def save_torch_state(state, destination, filename):
    if is_s3_uri(destination):
        bucket, key_prefix = split_s3_uri(destination)
        key = "/".join(part for part in [key_prefix, filename] if part)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or ".pt") as handle:
            temp_path = handle.name
        try:
            torch.save(state, temp_path)
            upload_file(temp_path, f"s3://{bucket}/{key}")
            return f"s3://{bucket}/{key}"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    path = os.path.join(destination, filename)
    ensure_parent_dir(path)
    torch.save(state, path)
    return path


def load_torch_state(source, map_location=None):
    if is_s3_uri(source):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as handle:
            temp_path = handle.name
        try:
            download_file(source, temp_path)
            return torch.load(temp_path, map_location=map_location)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return torch.load(source, map_location=map_location)


def upload_file(local_path, destination_uri):
    if not is_s3_uri(destination_uri):
        ensure_parent_dir(destination_uri)
        shutil.copy2(local_path, destination_uri)
        return destination_uri

    if boto3 is None:
        raise ImportError("boto3 is required for S3 destinations")

    bucket, key = split_s3_uri(destination_uri)
    client = boto3.client("s3")
    client.upload_file(local_path, bucket, key)
    return destination_uri


def download_file(source_uri, local_path):
    if not is_s3_uri(source_uri):
        ensure_parent_dir(local_path)
        shutil.copy2(source_uri, local_path)
        return local_path

    if boto3 is None:
        raise ImportError("boto3 is required for S3 destinations")

    bucket, key = split_s3_uri(source_uri)
    ensure_parent_dir(local_path)
    client = boto3.client("s3")
    client.download_file(bucket, key, local_path)
    return local_path