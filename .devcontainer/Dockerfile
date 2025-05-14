# Use the official Python 3.11 slim image as the base
FROM python:3.11-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USER=devuser
ARG UID=1000
ARG GID=1000
RUN groupadd --gid $GID $USER && \
    useradd --uid $UID --gid $GID -m $USER

# Set working directory
WORKDIR /workspace

# Switch to non-root user
USER $USER
