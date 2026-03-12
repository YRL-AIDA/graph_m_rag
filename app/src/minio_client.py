import logging
import os
from typing import BinaryIO, List, Any, Union
import io
from minio import Minio
from minio.error import S3Error
from app.config.settings import settings

class MinioClient:
    def __init__(self, logger: logging.Logger = None) -> None:
        endpoint = settings.s3_endpoint_clean
        access_key = settings.s3.S3_ACCESS_KEY
        secret_key = settings.s3.S3_SECRET_KEY
        secure = settings.s3_secure

        self.logger = logger or logging.getLogger(__name__)
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

        # Set bucket name from environment or default
        self.bucket_name = os.environ.get("S3_BUCKET_NAME", "pdf-processing")

        # Ensure bucket exists
        self._ensure_bucket_exists()

        self.logger.info(f'MinioClient initialized with endpoint: "{endpoint}", bucket: "{self.bucket_name}"')

    def _ensure_bucket_exists(self):
        """Ensure the bucket exists, create if it doesn't"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.logger.info(f"Creating bucket {self.bucket_name}")
                self.client.make_bucket(self.bucket_name)
                self.logger.info(f"Bucket {self.bucket_name} created successfully")
            else:
                self.logger.info(f"Bucket {self.bucket_name} already exists")
        except S3Error as e:
            self.logger.error(f"Failed to check/create bucket {self.bucket_name}: {e}")
            raise

    def list_buckets(self):
        """List all buckets"""
        try:
            buckets = self.client.list_buckets()
            return [{'Name': bucket.name, 'CreationDate': bucket.creation_date} for bucket in buckets]
        except S3Error as e:
            self.logger.error(f"Error listing buckets: {e}")
            raise

    def list_objects(self, bucket_name: str = None) -> List:
        """List objects in a bucket"""
        if bucket_name is None:
            bucket_name = self.bucket_name

        try:
            objects = []
            for obj in self.client.list_objects(bucket_name, recursive=True):
                objects.append({
                    'Key': obj.object_name,
                    'LastModified': obj.last_modified,
                    'Size': obj.size,
                    'ETag': obj.etag
                })
            return objects
        except S3Error as e:
            self.logger.error(f"Error listing objects in bucket {bucket_name}: {e}")
            raise

    def upload(self, bucket_name: str = None,
               object_name: str = None,
               data: Union[bytes, str, BinaryIO] = None,
               content_type: str = "application/octet-stream") -> None:
        """Upload data to MinIO"""
        if bucket_name is None:
            bucket_name = self.bucket_name

        if object_name is None:
            raise ValueError("object_name is required")

        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            elif hasattr(data, 'read'):
                data = data.read()

            if isinstance(data, bytes):
                data_stream = io.BytesIO(data)
                length = len(data)
            else:
                raise TypeError("Data must be bytes, str, or a file-like object")

            self.logger.info(f"MinioClient upload data to bucket {bucket_name} with object_name {object_name}")

            self.client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=data_stream,
                length=length,
                content_type=content_type
            )
            self.logger.info(f"Successfully uploaded {object_name} to {bucket_name}")
        except S3Error as e:
            self.logger.error(f"Error uploading {object_name} to {bucket_name}: {e}")
            raise

    def put_object(self, bucket_name: str = None,
                   object_name: str = None,
                   data: Union[bytes, str] = None,
                   content_type: str = "application/octet-stream") -> None:
        """Alias for upload method"""
        self.upload(bucket_name, object_name, data, content_type)

    def download(self, bucket_name: str = None, object_name: str = None) -> bytes:
        """Download object from MinIO and return as bytes"""
        if bucket_name is None:
            bucket_name = self.bucket_name

        if object_name is None:
            raise ValueError("object_name is required")

        try:
            self.logger.info(f"MinioClient downloading {object_name} from {bucket_name}")

            response = self.client.get_object(bucket_name, object_name)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except S3Error as e:
            self.logger.error(f"Error downloading {object_name} from {bucket_name}: {e}")
            raise

    def get_object(self, bucket_name: str = None, object_name: str = None) -> bytes:
        """Alias for download method"""
        return self.download(bucket_name, object_name)

    def object_exists(self, bucket_name: str = None, object_name: str = None) -> bool:
        """Check if object exists in MinIO"""
        if bucket_name is None:
            bucket_name = self.bucket_name

        if object_name is None:
            raise ValueError("object_name is required")

        try:
            self.client.stat_object(bucket_name, object_name)
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
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
            objects = self.client.list_objects(bucket_name, recursive=True)
            for obj in objects:
                self.client.remove_object(bucket_name, obj.object_name)

            # Then delete the bucket itself
            self.client.remove_bucket(bucket_name)
            self.logger.info(f"Bucket {bucket_name} deleted successfully")
        except S3Error as e:
            self.logger.error(f"Error removing bucket {bucket_name}: {e}")
            raise

    def remove_object(self, bucket_name: str = None, object_name: str = None) -> None:
        """Remove object from bucket"""
        if bucket_name is None:
            bucket_name = self.bucket_name

        if object_name is None:
            raise ValueError("object_name is required")

        try:
            self.client.remove_object(bucket_name, object_name)
            self.logger.info(f"Object {object_name} deleted from {bucket_name}")
        except S3Error as e:
            self.logger.error(f"Error removing object {object_name} from {bucket_name}: {e}")
            raise

    def get_presigned_url(self, bucket_name: str = None, object_name: str = None, expiration: int = 3600) -> str:
        """Generate presigned URL for object download"""
        if bucket_name is None:
            bucket_name = self.bucket_name

        if object_name is None:
            raise ValueError("object_name is required")

        try:
            url = self.client.presigned_get_object(
                bucket_name,
                object_name,
                expires=expiration
            )
            return url
        except S3Error as e:
            self.logger.error(f"Error generating presigned URL for {object_name}: {e}")
            raise