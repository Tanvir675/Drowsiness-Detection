# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required to compile dlib and OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port Streamlit uses
EXPOSE 8508

# Command to run your Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8508", "--server.address=0.0.0.0"]
