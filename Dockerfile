FROM tensorflow/tensorflow:latest-gpu-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y --no-install-recommends python3-opencv
RUN pip3 install pandas numpy scipy matplotlib keras

RUN mkdir maxime
RUN cd maxime
RUN git clone git@github.com:icannos/multiagent-particle-envs.git
RUN pip install -e 
COPY . app/
