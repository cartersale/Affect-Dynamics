# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (including build tools if needed for numpy/pandas)
# apt-get install -y build-essential may be needed for some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependency specification first for better caching
COPY pyproject.toml .
# If you have a requirements.txt, copy it too. 
# But this project seems to use pyproject.toml (setup.py style, likely setuptools/flit/poetry)
# Based on file list, there is a pyproject.toml.

# Install dependencies
# We install the package in editable mode or just install dependencies first
# Assuming standard setuptools or similar from pyproject.toml
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy the rest of the application code
COPY . .

# Install the package itself (if not already covered by `pip install .` above)
RUN pip install --no-cache-dir -e .

# Define environment variables if needed
# ENV MY_VAR=value

# By default, run the help command for the makefile or a bash shell
CMD ["bash"]
