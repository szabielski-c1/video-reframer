import boto3
import aiofiles
import asyncio
from botocore.exceptions import ClientError, NoCredentialsError, ConnectionError, EndpointConnectionError
from urllib.parse import urlparse, quote, unquote
import os
import logging
from typing import Optional
import tempfile
import time
import random

from app.config import settings

logger = logging.getLogger(__name__)

class S3Service:
    """Service for handling S3 file operations"""

    def __init__(self):
        self.s3_client = None
        self.default_bucket = settings.S3_BUCKET
        self.prefix = settings.S3_PREFIX
        self._initialize_client()

    async def _retry_with_backoff(self, operation, max_retries=3, base_delay=1.0):
        """Execute operation with exponential backoff retry logic"""
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return await operation()
            except (ConnectionError, EndpointConnectionError, ClientError) as e:
                last_exception = e

                # Don't retry on certain errors
                if isinstance(e, ClientError):
                    error_code = e.response.get('Error', {}).get('Code', '')
                    # Don't retry on authentication or permission errors
                    if error_code in ['InvalidAccessKeyId', 'SignatureDoesNotMatch', 'AccessDenied']:
                        raise e

                if attempt < max_retries:
                    # Calculate delay with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"S3 operation failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}. Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"S3 operation failed after {max_retries + 1} attempts: {str(e)}")

        raise last_exception

    def _initialize_client(self):
        """Initialize S3 client with credentials"""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            logger.info("S3 client initialized successfully")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise Exception("AWS credentials not configured")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise

    def _add_prefix(self, s3_key: str) -> str:
        """Add configured prefix to S3 key"""
        if self.prefix and not s3_key.startswith(self.prefix):
            # Ensure prefix ends with / and key doesn't start with /
            prefix = self.prefix.rstrip('/') + '/'
            key = s3_key.lstrip('/')
            return prefix + key
        return s3_key

    async def download_file(self, s3_url: str, local_path: str) -> str:
        """Download file from S3 to local path"""

        try:
            bucket, key = self.parse_s3_url(s3_url)

            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Download file using asyncio thread
            await asyncio.to_thread(
                self.s3_client.download_file,
                bucket,
                key,
                local_path
            )

            logger.info(f"Downloaded {s3_url} to {local_path}")
            return local_path

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise FileNotFoundError(f"S3 object not found: {s3_url}")
            elif error_code == 'NoSuchBucket':
                raise ValueError(f"S3 bucket not found: {bucket}")
            else:
                raise Exception(f"S3 download error: {error_code}")

        except Exception as e:
            logger.error(f"Failed to download {s3_url}: {str(e)}")
            raise

    async def upload_file(
        self,
        local_path: str,
        s3_key: str,
        bucket: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> str:
        """Upload local file to S3 with retry logic and smart multipart handling"""

        bucket_name = bucket or self.default_bucket
        if not bucket_name:
            raise ValueError("No S3 bucket specified")

        # Add prefix to key
        full_s3_key = self._add_prefix(s3_key)

        # Check file size to determine upload method
        file_size = os.path.getsize(local_path)
        large_file_threshold = 100 * 1024 * 1024  # 100MB

        logger.info(f"Uploading {local_path} ({file_size / (1024*1024):.1f}MB) to S3: {full_s3_key}")

        if file_size > large_file_threshold:
            # Use robust multipart upload for large files
            return await self.upload_large_file_with_retry(local_path, full_s3_key, bucket_name, content_type)
        else:
            # Use simple upload with retry for smaller files
            return await self.upload_small_file_with_retry(local_path, full_s3_key, bucket_name, content_type)

    async def upload_small_file_with_retry(self, local_path: str, s3_key: str, bucket_name: str, content_type: Optional[str] = None) -> str:
        """Upload small file with retry logic"""

        # Auto-detect content type if not provided
        if not content_type:
            content_type = self.get_content_type(local_path)

        # Prepare upload args
        extra_args = {
            'ContentType': content_type,
            'ACL': 'public-read'  # Make videos publicly accessible
        }

        async def upload_operation():
            await asyncio.to_thread(
                self.s3_client.upload_file,
                local_path,
                bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )

        try:
            await self._retry_with_backoff(upload_operation, max_retries=3)

            # Generate public URL
            s3_url = f"https://{bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{quote(s3_key)}"
            logger.info(f"Successfully uploaded {local_path} to {s3_url}")
            return s3_url

        except Exception as e:
            logger.error(f"Failed to upload {local_path} after retries: {str(e)}")
            raise Exception(f"Upload failed: {str(e)}")

    async def upload_large_file_with_retry(self, local_path: str, s3_key: str, bucket_name: str, content_type: Optional[str] = None) -> str:
        """Upload large file using multipart upload with retry logic"""

        if not content_type:
            content_type = self.get_content_type(local_path)

        file_size = os.path.getsize(local_path)
        chunk_size = 10 * 1024 * 1024  # 10MB chunks
        total_chunks = (file_size + chunk_size - 1) // chunk_size

        logger.info(f"Starting multipart upload for {local_path} ({total_chunks} chunks)")

        upload_id = None
        uploaded_parts = []

        try:
            # Initialize multipart upload with retry
            async def create_upload():
                return await asyncio.to_thread(
                    self.s3_client.create_multipart_upload,
                    Bucket=bucket_name,
                    Key=s3_key,
                    ContentType=content_type,
                    ACL='public-read'
                )

            response = await self._retry_with_backoff(create_upload)
            upload_id = response['UploadId']
            logger.info(f"Created multipart upload with ID: {upload_id}")

            # Upload each part with retry
            with open(local_path, 'rb') as file:
                for part_number in range(1, total_chunks + 1):
                    chunk_data = file.read(chunk_size)
                    if not chunk_data:
                        break

                    async def upload_part():
                        return await asyncio.to_thread(
                            self.s3_client.upload_part,
                            Bucket=bucket_name,
                            Key=s3_key,
                            PartNumber=part_number,
                            UploadId=upload_id,
                            Body=chunk_data
                        )

                    part_response = await self._retry_with_backoff(upload_part, max_retries=5)
                    uploaded_parts.append({
                        'ETag': part_response['ETag'],
                        'PartNumber': part_number
                    })

                    progress = (part_number / total_chunks) * 100
                    logger.info(f"Uploaded part {part_number}/{total_chunks} ({progress:.1f}%)")

            # Complete multipart upload with retry
            async def complete_upload():
                return await asyncio.to_thread(
                    self.s3_client.complete_multipart_upload,
                    Bucket=bucket_name,
                    Key=s3_key,
                    UploadId=upload_id,
                    MultipartUpload={'Parts': uploaded_parts}
                )

            await self._retry_with_backoff(complete_upload)

            s3_url = f"https://{bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{quote(s3_key)}"
            logger.info(f"Successfully completed multipart upload: {s3_url}")
            return s3_url

        except Exception as e:
            logger.error(f"Multipart upload failed: {str(e)}")

            # Attempt to abort multipart upload
            if upload_id:
                try:
                    await asyncio.to_thread(
                        self.s3_client.abort_multipart_upload,
                        Bucket=bucket_name,
                        Key=s3_key,
                        UploadId=upload_id
                    )
                    logger.info(f"Aborted multipart upload: {upload_id}")
                except Exception as abort_error:
                    logger.warning(f"Failed to abort multipart upload: {abort_error}")

            raise Exception(f"Large file upload failed: {str(e)}")

    async def upload_file_with_progress(
        self,
        local_path: str,
        s3_key: str,
        bucket: Optional[str] = None,
        progress_callback=None
    ) -> str:
        """Upload file with progress tracking and retry logic"""

        bucket_name = bucket or self.default_bucket
        if not bucket_name:
            raise ValueError("No S3 bucket specified")

        # Add prefix to key
        full_s3_key = self._add_prefix(s3_key)

        file_size = os.path.getsize(local_path)
        large_file_threshold = 100 * 1024 * 1024  # 100MB

        logger.info(f"Uploading with progress {local_path} ({file_size / (1024*1024):.1f}MB) to S3: {full_s3_key}")

        if file_size > large_file_threshold:
            # Use custom multipart upload with progress for large files
            return await self.upload_large_file_with_progress(local_path, full_s3_key, bucket_name, progress_callback)
        else:
            # Use boto3 transfer manager with progress for smaller files
            return await self.upload_small_file_with_progress(local_path, full_s3_key, bucket_name, progress_callback)

    async def upload_small_file_with_progress(self, local_path: str, s3_key: str, bucket_name: str, progress_callback=None) -> str:
        """Upload small file with progress tracking and retry"""

        file_size = os.path.getsize(local_path)
        content_type = self.get_content_type(local_path)

        # Create transfer config for multipart upload
        from boto3.s3.transfer import TransferConfig

        config = TransferConfig(
            multipart_threshold=1024 * 25,  # 25MB
            max_concurrency=10,
            multipart_chunksize=1024 * 25,
            use_threads=True
        )

        # Progress callback wrapper
        def progress_wrapper(bytes_transferred):
            if progress_callback:
                percentage = (bytes_transferred / file_size) * 100
                asyncio.create_task(progress_callback(percentage))

        async def upload_operation():
            await asyncio.to_thread(
                self.s3_client.upload_file,
                local_path,
                bucket_name,
                s3_key,
                Config=config,
                Callback=progress_wrapper,
                ExtraArgs={
                    'ContentType': content_type,
                    'ACL': 'public-read'
                }
            )

        try:
            await self._retry_with_backoff(upload_operation, max_retries=3)

            s3_url = f"https://{bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{quote(s3_key)}"
            logger.info(f"Successfully uploaded with progress: {s3_url}")
            return s3_url

        except Exception as e:
            logger.error(f"Upload with progress failed: {str(e)}")
            raise

    async def upload_large_file_with_progress(self, local_path: str, s3_key: str, bucket_name: str, progress_callback=None) -> str:
        """Upload large file with detailed progress tracking and retry logic"""

        content_type = self.get_content_type(local_path)
        file_size = os.path.getsize(local_path)
        chunk_size = 10 * 1024 * 1024  # 10MB chunks
        total_chunks = (file_size + chunk_size - 1) // chunk_size

        logger.info(f"Starting multipart upload with progress for {local_path} ({total_chunks} chunks)")

        upload_id = None
        uploaded_parts = []
        total_uploaded = 0

        try:
            # Initialize multipart upload with retry
            async def create_upload():
                return await asyncio.to_thread(
                    self.s3_client.create_multipart_upload,
                    Bucket=bucket_name,
                    Key=s3_key,
                    ContentType=content_type,
                    ACL='public-read'
                )

            response = await self._retry_with_backoff(create_upload)
            upload_id = response['UploadId']
            logger.info(f"Created multipart upload with ID: {upload_id}")

            # Upload each part with retry and progress
            with open(local_path, 'rb') as file:
                for part_number in range(1, total_chunks + 1):
                    chunk_data = file.read(chunk_size)
                    if not chunk_data:
                        break

                    async def upload_part():
                        return await asyncio.to_thread(
                            self.s3_client.upload_part,
                            Bucket=bucket_name,
                            Key=s3_key,
                            PartNumber=part_number,
                            UploadId=upload_id,
                            Body=chunk_data
                        )

                    part_response = await self._retry_with_backoff(upload_part, max_retries=5)
                    uploaded_parts.append({
                        'ETag': part_response['ETag'],
                        'PartNumber': part_number
                    })

                    total_uploaded += len(chunk_data)
                    progress_percent = (total_uploaded / file_size) * 100

                    # Call progress callback
                    if progress_callback:
                        await progress_callback(progress_percent)

                    logger.info(f"Uploaded part {part_number}/{total_chunks} ({progress_percent:.1f}%)")

            # Complete multipart upload with retry
            async def complete_upload():
                return await asyncio.to_thread(
                    self.s3_client.complete_multipart_upload,
                    Bucket=bucket_name,
                    Key=s3_key,
                    UploadId=upload_id,
                    MultipartUpload={'Parts': uploaded_parts}
                )

            await self._retry_with_backoff(complete_upload)

            # Final progress callback
            if progress_callback:
                await progress_callback(100.0)

            s3_url = f"https://{bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{quote(s3_key)}"
            logger.info(f"Successfully completed multipart upload with progress: {s3_url}")
            return s3_url

        except Exception as e:
            logger.error(f"Multipart upload with progress failed: {str(e)}")

            # Attempt to abort multipart upload
            if upload_id:
                try:
                    await asyncio.to_thread(
                        self.s3_client.abort_multipart_upload,
                        Bucket=bucket_name,
                        Key=s3_key,
                        UploadId=upload_id
                    )
                    logger.info(f"Aborted multipart upload: {upload_id}")
                except Exception as abort_error:
                    logger.warning(f"Failed to abort multipart upload: {abort_error}")

            raise Exception(f"Large file upload with progress failed: {str(e)}")

    async def generate_presigned_url(
        self,
        s3_key: str,
        bucket: Optional[str] = None,
        expiration: int = 3600
    ) -> str:
        """Generate presigned URL for private access"""

        bucket_name = bucket or self.default_bucket

        try:
            url = await asyncio.to_thread(
                self.s3_client.generate_presigned_url,
                'get_object',
                Params={'Bucket': bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )

            return url

        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {str(e)}")
            raise

    async def check_file_exists(self, s3_url: str) -> bool:
        """Check if file exists in S3"""

        try:
            bucket, key = self.parse_s3_url(s3_url)

            await asyncio.to_thread(
                self.s3_client.head_object,
                Bucket=bucket,
                Key=key
            )

            return True

        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise

    async def get_file_metadata(self, s3_url: str) -> dict:
        """Get file metadata from S3"""

        try:
            bucket, key = self.parse_s3_url(s3_url)

            response = await asyncio.to_thread(
                self.s3_client.head_object,
                Bucket=bucket,
                Key=key
            )

            return {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'content_type': response.get('ContentType', 'unknown'),
                'etag': response['ETag'].strip('"')
            }

        except ClientError as e:
            logger.error(f"Failed to get metadata for {s3_url}: {e}")
            raise

    async def delete_file(self, s3_url: str) -> bool:
        """Delete file from S3"""

        try:
            bucket, key = self.parse_s3_url(s3_url)

            await asyncio.to_thread(
                self.s3_client.delete_object,
                Bucket=bucket,
                Key=key
            )

            logger.info(f"Deleted {s3_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete {s3_url}: {str(e)}")
            return False

    async def copy_file(self, source_url: str, dest_key: str, dest_bucket: Optional[str] = None) -> str:
        """Copy file within S3"""

        dest_bucket_name = dest_bucket or self.default_bucket
        source_bucket, source_key = self.parse_s3_url(source_url)

        try:
            copy_source = {'Bucket': source_bucket, 'Key': source_key}

            await asyncio.to_thread(
                self.s3_client.copy_object,
                CopySource=copy_source,
                Bucket=dest_bucket_name,
                Key=dest_key,
                ACL='public-read'
            )

            dest_url = f"https://{dest_bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{quote(dest_key)}"
            return dest_url

        except Exception as e:
            logger.error(f"Failed to copy {source_url} to {dest_key}: {str(e)}")
            raise

    def parse_s3_url(self, s3_url: str) -> tuple[str, str]:
        """Parse S3 URL to extract bucket and key"""

        if s3_url.startswith('s3://'):
            # s3://bucket/key format
            parsed = urlparse(s3_url)
            bucket = parsed.netloc
            key = unquote(parsed.path.lstrip('/'))
        elif 's3.amazonaws.com' in s3_url or '.s3.' in s3_url:
            # https://bucket.s3.region.amazonaws.com/key format
            parsed = urlparse(s3_url)
            if '.s3.' in parsed.netloc:
                bucket = parsed.netloc.split('.s3.')[0]
            else:
                # Old format: https://s3.amazonaws.com/bucket/key
                path_parts = parsed.path.lstrip('/').split('/', 1)
                bucket = path_parts[0]
                key = unquote(path_parts[1] if len(path_parts) > 1 else '')
                return bucket, key

            key = unquote(parsed.path.lstrip('/'))
        else:
            raise ValueError(f"Invalid S3 URL format: {s3_url}")

        if not bucket or not key:
            raise ValueError(f"Could not parse bucket/key from URL: {s3_url}")

        return bucket, key

    def get_content_type(self, file_path: str) -> str:
        """Determine content type from file extension"""

        extension = os.path.splitext(file_path)[1].lower()

        content_types = {
            '.mp4': 'video/mp4',
            '.avi': 'video/avi',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.json': 'application/json',
            '.txt': 'text/plain'
        }

        return content_types.get(extension, 'application/octet-stream')

    async def create_multipart_upload(self, s3_key: str, bucket: Optional[str] = None) -> str:
        """Initialize multipart upload for large files"""

        bucket_name = bucket or self.default_bucket

        try:
            response = await asyncio.to_thread(
                self.s3_client.create_multipart_upload,
                Bucket=bucket_name,
                Key=s3_key,
                ACL='public-read'
            )

            return response['UploadId']

        except Exception as e:
            logger.error(f"Failed to create multipart upload: {str(e)}")
            raise

    async def upload_part(
        self,
        upload_id: str,
        part_number: int,
        data: bytes,
        s3_key: str,
        bucket: Optional[str] = None
    ) -> dict:
        """Upload a part of multipart upload"""

        bucket_name = bucket or self.default_bucket

        try:
            response = await asyncio.to_thread(
                self.s3_client.upload_part,
                Bucket=bucket_name,
                Key=s3_key,
                PartNumber=part_number,
                UploadId=upload_id,
                Body=data
            )

            return {
                'ETag': response['ETag'],
                'PartNumber': part_number
            }

        except Exception as e:
            logger.error(f"Failed to upload part {part_number}: {str(e)}")
            raise

    async def complete_multipart_upload(
        self,
        upload_id: str,
        parts: list,
        s3_key: str,
        bucket: Optional[str] = None
    ) -> str:
        """Complete multipart upload"""

        bucket_name = bucket or self.default_bucket

        try:
            await asyncio.to_thread(
                self.s3_client.complete_multipart_upload,
                Bucket=bucket_name,
                Key=s3_key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )

            s3_url = f"https://{bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{quote(s3_key)}"
            return s3_url

        except Exception as e:
            logger.error(f"Failed to complete multipart upload: {str(e)}")
            raise

    async def abort_multipart_upload(
        self,
        upload_id: str,
        s3_key: str,
        bucket: Optional[str] = None
    ):
        """Abort multipart upload"""

        bucket_name = bucket or self.default_bucket

        try:
            await asyncio.to_thread(
                self.s3_client.abort_multipart_upload,
                Bucket=bucket_name,
                Key=s3_key,
                UploadId=upload_id
            )

        except Exception as e:
            logger.warning(f"Failed to abort multipart upload: {str(e)}")

    async def list_files(self, prefix: str = "", bucket: Optional[str] = None) -> list:
        """List files in S3 bucket with optional prefix"""

        bucket_name = bucket or self.default_bucket

        try:
            response = await asyncio.to_thread(
                self.s3_client.list_objects_v2,
                Bucket=bucket_name,
                Prefix=prefix,
                MaxKeys=1000
            )

            files = []
            for obj in response.get('Contents', []):
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'url': f"https://{bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{quote(obj['Key'])}"
                })

            return files

        except Exception as e:
            logger.error(f"Failed to list files: {str(e)}")
            raise