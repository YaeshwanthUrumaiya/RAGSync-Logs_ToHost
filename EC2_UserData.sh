#!/bin/bash

# Start with sudo
sudo -i

# Update package lists and upgrade existing packages
sudo apt-get update -y 
sudo apt-get upgrade -y

# Install required packages
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update -y
sudo apt-get install -y python3.9 python3-pip python3.9-distutils sqlite3

# Clone the repository
git clone https://github.com/YaeshwanthUrumaiya/RAGSync-Logs_ToHost.git

# Copy the cloned contents to the current directory
cp -R RAGSync-Logs_ToHost/* /root

# Install requirements
pip install --no-cache-dir streamlit boto3 --break-system-packages
pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu --break-system-packages
pip install --no-cache-dir --no-deps mpmath urllib3 typing_extensions threadpoolctl sympy safetensors regex pyyaml Pillow packaging numpy networkx MarkupSafe joblib idna fsspec filelock colorama charset-normalizer certifi tqdm scipy requests jinja2 scikit-learn huggingface_hub tokenizers transformers sentence_transformers --break-system-packages
pip install --no-cache-dir faiss-cpu langchain langchain-aws langchain-community langchain-core rank-bm25 --break-system-packages

echo "Setup completed successfully!"
