#Finalized
import os
import sys
import boto3
import shutil
import argparse
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    hf_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs, show_progress=True, cache_folder=main_file_path_embedding+'/'
    )
    print("Embeddings Setup Done")
    return hf_embeddings

def SetupVectorDatabase(files_path,vb_path,hf_embeddings):
    loader = DirectoryLoader(
        files_path,
        glob="*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, is_separator_regex=False, separators="\n")
    splits = text_splitter.split_documents(documents) #Splits the files into chunks.
    print("Files read and split! (RD)")
    
    vectorstore = FAISS.from_documents(documents=splits, embedding=hf_embeddings) #This is to create the database from scratch.
    vectorstore.save_local(vb_path)
    print("Process done for Files")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--doctype", type=str, default="RD", help="SD for SupportingDoc; RD for RawDocs")
    parser.add_argument("--endbucketname", type=str, help="FinalBucketname")
    parser.add_argument("--embedmodel", type=str, default="BAAI/bge-small-en", help="SD for SupportingDoc; RD for RawDocs")
    
    args = parser.parse_args()
    
    files_path = "./SupportingDoc" if args.doctype == "SD" else "./RawDoc"
    vb_path = "./FAISS_SDdata" if args.doctype == "SD" else "./FAISS_data"
    ensure_folder_exists(vb_path)
    model_name  = args.embedmodel
    hf_embeddings = SetupEmbedding(model_name)
    SetupVectorDatabase(files_path,vb_path,hf_embeddings)
    Bucket_Name = args.endbucketname
    s3 = boto3.client("s3") 
    s3.upload_file(Filename=vb_path + "/" + "index.faiss", Bucket=Bucket_Name, Key=vb_path[2:]+"/index.faiss")
    s3.upload_file(Filename=vb_path + "/" + "index.pkl", Bucket=Bucket_Name, Key=vb_path[2:]+"/index.pkl")