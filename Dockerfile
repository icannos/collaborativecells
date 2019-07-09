FROM tensorflow/tensorflow:latest-gpu-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y --no-install-recommends
RUN apt-get install -y --no-install-recommends python3-opencv git
RUN pip3 install pandas numpy scipy matplotlib keras

RUN mkdir maxime
RUN cd maxime
RUN git clone https://github.com/icannos/multiagent-particle-envs.git
RUN cd ..
COPY . app/
