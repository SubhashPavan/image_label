FROM python:3.6

MAINTAINER SubhashPavan "pavansubhash@gmail.com"

RUN apt-get update -y

RUN apt-get install -y python3-pip
RUN apt-get install -y libsm6 libxext6 libxrender-dev

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]
