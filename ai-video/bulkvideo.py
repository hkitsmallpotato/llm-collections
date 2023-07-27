import boto3
import os, json, subprocess, time

# Initialize and setup
print("Initialize...")

s3_keyid = os.environ['S3_KEYID']
s3_secret = os.environ['S3_SECRET']
s3_host = os.environ['S3_HOST']
s3_bucket = os.environ['S3_BUCKET']
s3_basepath = os.environ['S3_BASEPATH']
session = boto3.Session(aws_access_key_id=s3_keyid, aws_secret_access_key=s3_secret)

s3_client = session.resource('s3', endpoint_url=s3_host)
bucket = s3_client.Bucket(s3_bucket)

# Read request file
print("Read request file from S3...")

original_filename = "bulk_request.json"
bucket.download_file(s3_basepath + "/" + original_filename, original_filename)
bulk_request = None
with open(original_filename, "r") as f:
    bulk_request = json.load(f)
num_req = len(bulk_request)
print("OK. Total {n} requests.".format(n=num_req))

# Iterate generation
for idx, req in enumerate(bulk_request):
    try:
        print("---- Request {i} of {n}: prompt = {prompt}, file = {file}".format(
            i=idx+1, n=num_req, prompt=req['prompt'], file=req['file']))
        start_time = time.time()
        local_path = ""
        result = subprocess.run(
            [sys.executable, ""], 
            capture_output=True, text=True
        )
        print("stdout:\n", result.stdout)
        print("stderr:\n", result.stderr)
        gen_time = time.time()
        print("Uploading...")
        bucket.upload_file(local_path, s3_basepath + "/outs/" + req['file'])
        end_time = time.time()
        print("---- Performance Statistic: Generation time = {gen}, Upload time = {up}".format(
            gen = gen_time - start_time,
            up = end_time - gen_time
        ))
    except Exception as e:
        print(str(e))

print("Done.")
