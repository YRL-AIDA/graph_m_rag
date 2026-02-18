import logging
import os
from typing import BinaryIO, List, Tuple, Any

import boto3
from boto3_type_annotations.s3 import ObjectSummary


class S3Client:
    def __init__(self, logger: logging.Logger) -> None:
        endpoint = os.environ.get("S3_URL", "http://localhost:9001")
        access_key = os.environ.get("S3_ACCESS_KEY", "minio")
        secret_key = os.environ.get("S3_SECRET_KEY", "minio123")
        verify = "false" #os.environ.get("S3_VERIFY_TLS", "false").lower() == "true"

        self.logger = logger
        self.client = boto3.resource("s3", endpoint_url=endpoint, aws_access_key_id=access_key, aws_secret_access_key=secret_key, verify=verify)
        #logger.info(f'S3Client endpoint is "{endpoint}", verify is "{verify}"')

    def check_bucket(self, bucket_name: str) -> None:
        bucket = self.client.Bucket(bucket_name)
        if not bucket.creation_date:
            self.client.create_bucket(Bucket=bucket_name)

    def list_objects(self, bucket_name: str) -> List[ObjectSummary]:
        bucket = self.client.Bucket(bucket_name)

        if not bucket.creation_date:
            return []

        return [s3_object for s3_object in bucket.objects.all()]

    def list_buckets(self):
        return self.client.buckets.all()

    def upload(self, bucket_name: str, object_name: str, file: BinaryIO) -> None:
        self.logger.info(f"S3Client upload file to bucket {bucket_name} with object_name {object_name}")

        object_name = self.__check_filename_length(object_name)
        bucket = self.client.Bucket(bucket_name)
        bucket.put_object(Body=file, Key=object_name)

    def download(self, s3_path: str, output_dir: str) -> str:
        self.logger.info(f"S3Client download file from {s3_path}")

        bucket_name, object_name = self.__split_s3_path(s3_path)
        output_path = os.path.join(output_dir, object_name)
        bucket = self.client.Bucket(bucket_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        bucket.download_file(object_name, output_path)
        return output_path

    def remove_bucket(self, bucket_name: str) -> None:
        bucket = self.client.Bucket(bucket_name)

        if not bucket.creation_date:
            return

        bucket.objects.all().delete()
        bucket.delete()

    def remove_object(self, bucket_name: str, object_name: str) -> None:
        bucket = self.client.Bucket(bucket_name)
        s3_object = bucket.Object(object_name)
        s3_object.delete()

    def __split_s3_path(self, s3_path: str) -> Tuple[str, str]:
        list_dirs = s3_path.split("s3://")[-1].split("/")
        bucket_name = list_dirs[0]
        object_name = "/".join(list_dirs[1:])
        return bucket_name, object_name

    def __check_filename_length(self, filename: str, max_filename_length: int = 255) -> str:
        if len(filename) <= max_filename_length:
            return filename

        dirs = filename.split(".")
        name, ext = dirs[0], "." + ".".join(dirs[-2:]) if len(dirs) > 2 else os.path.splitext(filename)
        filename = f"{name[:max_filename_length - len(ext)]}{ext}"
        return filename
