import os, sys
import json
import random
import glob
import boto3
import ntpath
import numpy as np
from concurrent.futures.thread import ThreadPoolExecutor
import subprocess
# subprocess.run(["ls", "-l"])


#################################################################

s3_client = boto3.client('s3')
boto3.setup_default_session(profile_name='ph-ai-dev')  # To switch between different AWS accounts

def download_s3_file(s3_url, dst_dir='.', filename=None, silent=False, overwrite=False):
    filename = s3_url.split("/").pop()
    out_path = os.path.join(dst_dir, filename)
    if os.path.exists(out_path) and not overwrite:
        return out_path 
    try:
        if not silent:
            print(f'Downloading file from: {s3_url} to {out_path}')
        s3_url_components = s3_url.replace("s3://", "")
        bucket = s3_url_components.replace("s3://", "").split("/")[0]
        key = s3_url_components.replace(f'{bucket}/', "")
        with open(out_path, 'wb') as f:
            s3_client.download_fileobj(bucket, key, f)
        return out_path
    except Exception as e:
        print(f"Failed to download file from S3: {s3_url}")
        # raise e

def download_image(s3_url, dst_dir='.'):
    return download_s3_file(s3_url, dst_dir=dst_dir, silent=False)

def download_images(s3_urls, dst_dir='.'):
    executor = ThreadPoolExecutor(max_workers=64)
    for s3_url in s3_urls:
        executor.submit(download_image, s3_url, dst_dir)
    executor.shutdown(wait=True)
    
def load_list(fn):
    with open(fn) as f:
        lines = f.read().splitlines()
    return lines

def save_list(lst, fn):
    with open(fn, 'w') as f:
        for item in lst:
            f.write("%s\n" % item)

######################################

#  aws s3 ls s3://ai-labeling/FPL/cainspect/aug12/
s3bucket = 's3://ai-labeling/FPL/cainspect/aug12/'
# s3_urls = [f'{s3bucket}/{f}' for f in imgs]

# # s3_urls = load_list(S3FILE)
# download_images(s3_urls)
# print('Done.')
