FROM ubuntu:16.04
FROM continuumio/miniconda3
FROM tensorflow/tensorflow:latest-py3

MAINTAINER SubhashPavan "pavansubhash@gmail.com"

RUN apt-get update -y

#RUN apt-get install -y python3.5
#RUN apt-get install -y python3-pip
RUN apt-get install -y libsm6 libxext6 libxrender-dev


COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

#RUN conda install -c conda-forge keras
RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]
