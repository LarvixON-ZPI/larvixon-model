import boto3
from botocore.config import Config
import os

s3 = boto3.client(
    "s3",
    endpoint_url="https://s3min2.e-science.pl",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name="us-east-1",
)

print("Buckets:", [b["Name"] for b in s3.list_buckets()["Buckets"]])

resp = s3.list_objects_v2(Bucket="s3min-adam.junka-1744366756")
print("Objects:", [o["Key"] for o in resp.get("Contents", [])])
