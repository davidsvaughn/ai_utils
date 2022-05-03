import os
from glob import glob
import boto3
from concurrent.futures.thread import ThreadPoolExecutor

s3_client = boto3.client('s3')
boto3.setup_default_session(profile_name='ph-ai-dev')  ## <---- change this?

# img_path = '/home/david/code/phawk/data/solar/indivillage/images/'
# s3_path = 's3://ai-labeling/IndiVillage/Solar/images/'

img_path = '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/images/'
s3_path = 's3://ai-labeling/transmission/images/wood/'

# img_path = '/home/david/code/phawk/data/generic/transmission/master/images/'
# s3_path = 's3://ph-dvaughn-dev/transmission/data/master/images/'

#----------------------------------------------------------------------

def upload_s3_file(src_file, s3_path):
    filename = src_file.split("/").pop()
    try:
        s3_url_components = s3_path.replace("s3://", "")
        bucket = s3_url_components.split("/")[0]
        key = s3_url_components.replace(f'{bucket}/', "")
        key =  os.path.join(key, filename)
        with open(src_file, 'rb') as f:
            s3_client.upload_fileobj(f, bucket, key)
    except Exception as e:
        print(f"Failed to upload file to S3: {src_file}")

def upload_files(files, s3_path):
    executor = ThreadPoolExecutor(max_workers=64)
    for i,file_path in enumerate(files):
        executor.submit(upload_s3_file, file_path, s3_path)
        if i%100==0:
            print(f"{i}\t{file_path}")
    executor.shutdown(wait=True)


img_files = glob(os.path.join(img_path, '*.jpg'))
upload_files(img_files, s3_path)
print('Done uploading to S3.')