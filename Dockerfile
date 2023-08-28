FROM nvcr.io/nvidia/tritonserver:22.12-py3

RUN apt update && apt install ffmpeg libsm6 libxext6 -y