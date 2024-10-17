#Finalized
import os
import sys
import boto3
import shutil
import argparse
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def ensure_folder_exists(folder_path):
    main_file_path = folder_path
    if os.path.exists(main_file_path):
        print(f"Deleting existing folder: {main_file_path}")
        shutil.rmtree(main_file_path)
    print(f"Creating folder: {main_file_path}")
    os.makedirs(main_file_path, exist_ok=True)
    
def SetupEmbedding(model_name):
    main_file_path_embedding = f"./Embeddings"
    ensure_folder_exists(main_file_path_embedding)
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf_embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs, show_progress=True, cache_folder=main_file_path_embedding+'/'
    )
    print("Embeddings Setup Done")
    return hf_embeddings

def download_dir(client, resource, dist, local='/tmp', bucket='your_bucket'):
    paginator = client.get_paginator('list_objects')
    for result in paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=dist):
        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                download_dir(client, resource, subdir.get('Prefix'), local, bucket)
        for file in result.get('Contents', []):
            dest_pathname = os.path.join(local, file.get('Key'))
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
            if not file.get('Key').endswith('/'):
                resource.meta.client.download_file(bucket, file.get('Key'), dest_pathname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--bucketname", type=str, default="", help="s3 Bucket name")
    parser.add_argument("--embedmodel", type=str, default="BAAI/bge-small-en", help="SD for SupportingDoc; RD for RawDocs")
    parser.add_argument("--customacc", type=bool, default=False, help="Enable custom account")
    parser.add_argument("--accesskey", type=str, default="", help="Access key")
    parser.add_argument("--secretkey", type=str, default="", help="Secret key")
    
    args = parser.parse_args()
    
    if args.customacc == True:
        print("Using Custom User")
        aws_access_key = args.accesskey
        aws_secret_key = args.secretkey
    else:
        print("Using Default User")
        load_dotenv()
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    Bucket_Name = args.bucketname
    main_file_path = "./"
    client = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
    resource = boto3.resource('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
    download_dir(client, resource, '', main_file_path, bucket=Bucket_Name)
    model_name  = args.embedmodel
    hf_embeddings = SetupEmbedding(model_name)