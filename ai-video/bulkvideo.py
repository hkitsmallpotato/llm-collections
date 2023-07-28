import boto3
import os, sys, json, subprocess, time

from os import listdir
from os.path import isfile, join

def get_file_only(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]

# Initialize and setup
print("Initialize...")

s3_keyid = os.environ['S3_KEYID'].strip('\"')
s3_secret = os.environ['S3_SECRET'].strip('\"')
s3_host = os.environ['S3_HOST'].strip('\"')
s3_bucket = os.environ['S3_BUCKET'].strip('\"')
s3_basepath = os.environ['S3_BASEPATH'].strip('\"')
local_basepath = os.environ['LOCAL_BASEPATH'].strip('\"')

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
        prompt = req['prompt']
        negative_prompt = "text, watermark, copyright, blurry, low resolution, blur, low quality"
        output_path = local_basepath + "/output/" + req['file']
        #python inference.py -m "/content/model" -p {prompt} -n {negative} 
        # -W 576 -H 320 -o /content/outputs -d cuda -x -s {num_steps} 
        # -g {guidance_scale} -f {fps} -T {num_frames}
        result = subprocess.run(
            [sys.executable, "inference.py", "-m", local_basepath + "/model",
            "-p", prompt, "-n", negative_prompt, "-W", "576", "-H", "320",
            "-o", output_path,
            "-d", "cuda", "-x", "-s", "25", "-g", "23", 
            "-f", "10", "-T", "30"], 
            capture_output=True, text=True,
            cwd = local_basepath + "/Text-To-Video-Finetuning"
        )
        print("stdout:\n", result.stdout)
        print("stderr:\n", result.stderr)
        gen_time = time.time()
        # Need a hack to get the local file
        local_path = join(output_path, get_file_only(output_path)[0])
        print("[Debug: {lp}]".format(lp=local_path))
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
