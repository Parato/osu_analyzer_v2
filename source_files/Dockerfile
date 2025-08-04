# Dockerfile
#
# This file defines the container image that Vertex AI will use to run your code.
# It specifies the base image, copies your code, and installs dependencies.

# Use an official NVIDIA CUDA image as a base. This provides the GPU drivers.
# Choose a version compatible with your PyTorch version.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set the working directory inside the container
WORKDIR /app

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, and other necessary system libraries
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- KORREKTUR: Erstelle eine Verknüpfung, damit der Befehl 'python' auf 'python3' verweist ---
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all your Python scripts and utility files into the container
COPY . .

# This container can be used for both data generation and training.
# The entry point will be specified in the Vertex AI job submission command.
