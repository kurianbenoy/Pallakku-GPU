FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN apt-get update
RUN apt-get install -y ffmpeg git-lfs

RUN pip install faster-whisper

RUN git lfs install

RUN git clone https://huggingface.co/kurianbenoy/vegam-whisper-medium-ml

