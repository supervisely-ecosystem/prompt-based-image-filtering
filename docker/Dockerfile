FROM supervisely/base-py-sdk:6.69.70

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y build-essential
RUN apt-get install -y git ffmpeg libsm6 libxext6 ninja-build

RUN pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip3 install Pillow==9.2.0 \
    open-clip-torch==2.16.0

RUN pip3 install supervisely==6.70.10

# download weights for two models
RUN python3 docker/download_weights.py
