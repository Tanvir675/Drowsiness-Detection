FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake

RUN pip install --upgrade pip wheel setuptools

RUN pip wheel dlib==19.24.6 -w /wheels
