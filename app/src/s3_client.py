import logging
import os
from typing import BinaryIO, List, Tuple, Any, Union
import io

import boto3
from botocore.exceptions import ClientError


class S3Client:
    def __init__(self, logger: logging.Logger = None) -> None:
        endpoint = os.environ.get("S3_URL", "http://localhost:9001")
        access_key = os.environ.get("S3_ACCESS_KEY", "minio")
        secret_key = os.environ.get("S3_SECRET_KEY", "minio123")
        verify = os.environ.get("S3_VERIFY_TLS", "false").lower() != "true"

        self.logger = logger or logging.getLogger(__name__)
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            verify=verify
        )
        self.resource = boto3.resource(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            verify=verify
        )

        # Set bucket name from environment or default
        self.bucket_name = os.environ.get("S3_BUCKET_NAME", "pdf-processing")

        # Ensure bucket exists
        self._ensure_bucket_exists()

        self.logger.info(f'S3Client initialized with endpoint: "{endpoint}", bucket: "{self.bucket_name}"')

    def _ensure_bucket_exists(self):
        """Ensure the bucket exists, create if it doesn't"""
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
            self.logger.info(f"Bucket {self.bucket_name} already exists")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                self.logger.info(f"Creating bucket {self.bucket_name}")
                try:
                    self.client.create_bucket(Bucket=self.bucket_name)
                    self.logger.info(f"Bucket {self.bucket_name} created successfully")
                except ClientError as create_error:
                    self.logger.error(f"Failed to create bucket {self.bucket_name}: {create_error}")
                    raise
            else:
                self.logger.error(f"Error checking bucket {self.bucket_name}: {e}")
                raise

    def list_buckets(self):
        """List all buckets"""
        try:
            response = self.client.list_buckets()
            return response['Buckets']
        except ClientError as e:
            self.logger.error(f"Error listing buckets: {e}")
            raise

    def list_objects(self, bucket_name: str = None) -> List:
        """List objects in a bucket"""
        if bucket_name is None:
            bucket_name = self.bucket_name

        try:
            response = self.client.list_objects_v2(Bucket=bucket_name)
            return response.get('Contents', [])
        except ClientError as e:
            self.logger.error(f"Error listing objects in bucket {bucket_name}: {e}")
            raise

    def upload(self, bucket_name: str = None,
               object_name: str = None,
               data: Union[bytes, str, BinaryIO] = None,
               content_type: str = "application/octet-stream") -> None:
        """Upload data to S3"""
        if bucket_name is None:
            bucket_name = self.bucket_name

        if object_name is None:
            raise ValueError("object_name is required")

        try:
            if isinstance(data, str):
                data = data.encode('utf-8')

            self.logger.info(f"S3Client upload data to bucket {bucket_name} with object_name {object_name}")

            self.client.put_object(
                Bucket=bucket_name,
                Key=object_name,
                Body=data,
                ContentType=content_type
            )
            self.logger.info(f"Successfully uploaded {object_name} to {bucket_name}")
        except ClientError as e:
            self.logger.error(f"Error uploading {object_name} to {bucket_name}: {e}")
            raise

    def put_object(self, bucket_name: str = None,
                   object_name: str = None,
                   data: Union[bytes, str] = None,
                   content_type: str = "application/octet-stream") -> None:
        """Alias for upload method"""
        self.upload(bucket_name, object_name, data, content_type)

    def download(self, bucket_name: str = None, object_name: str = None) -> bytes:
        """Download object from S3 and return as bytes"""
        if bucket_name is None:
            bucket_name = self.bucket_name

        if object_name is None:
            raise ValueError("object_name is required")

        try:
            self.logger.info(f"S3Client downloading {object_name} from {bucket_name}")

            response = self.client.get_object(Bucket=bucket_name, Key=object_name)
            return response['Body'].read()
        except ClientError as e:
            self.logger.error(f"Error downloading {object_name} from {bucket_name}: {e}")
            raise

    def get_object(self, bucket_name: str = None, object_name: str = None) -> bytes:
        """Alias for download method"""
        return self.download(bucket_name, object_name)

    def object_exists(self, bucket_name: str = None, object_name: str = None) -> bool:
        """Check if object exists in S3"""
        if bucket_name is None:
            bucket_name = self.bucket_name

        if object_name is None:
            raise ValueError("object_name is required")

        try:
            self.client.head_object(Bucket=bucket_name, Key=object_name)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                self.logger.error(f"Error checking if object exists {object_name} in {bucket_name}: {e}")
                raise

    def remove_bucket(self, bucket_name: str = None) -> None:
        """Remove bucket and all its contents"""
        if bucket_name is None:
            bucket_name = self.bucket_name

        try:
            # First delete all objects in the bucket
            bucket = self.resource.Bucket(bucket_name)
            bucket.objects.all().delete()

            # Then delete the bucket itself
            bucket.delete()
            self.logger.info(f"Bucket {bucket_name} deleted successfully")
        except ClientError as e:
            self.logger.error(f"Error removing bucket {bucket_name}: {e}")
            raise

    def remove_object(self, bucket_name: str = None, object_name: str = None) -> None:
        """Remove object from bucket"""
        if bucket_name is None:
            bucket_name = self.bucket_name

        if object_name is None:
            raise ValueError("object_name is required")

        try:
            self.client.delete_object(Bucket=bucket_name, Key=object_name)
            self.logger.info(f"Object {object_name} deleted from {bucket_name}")
        except ClientError as e:
            self.logger.error(f"Error removing object {object_name} from {bucket_name}: {e}")
            raise

    def get_presigned_url(self, bucket_name: str = None, object_name: str = None, expiration: int = 3600) -> str:
        """Generate presigned URL for object download"""
        if bucket_name is None:
            bucket_name = self.bucket_name

        if object_name is None:
            raise ValueError("object_name is required")

        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': object_name},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            self.logger.error(f"Error generating presigned URL for {object_name}: {e}")
            raise