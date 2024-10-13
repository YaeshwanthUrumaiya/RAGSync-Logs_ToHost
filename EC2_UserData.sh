#!/bin/bash

# Start with sudo
# sudo -i

# Update package lists and upgrade existing packages
apt-get update -y && apt-get upgrade -y

# Install required packages
apt-get install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update -y
apt-get install -y python3.9 python3-pip python3.9-distutils sqlite3

# Clone the repository
git clone https://github.com/YaeshwanthUrumaiya/RAGSync-Logs_ToHost.git

# Copy the cloned contents to the current directory
cp -R RAGSync-Logs_ToHost/* .

# Install requirements
pip install -r requirements.txt --break-system-packages
