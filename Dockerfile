#FROM ubuntu:20.04
FROM nvidia/cuda:11.7.0-base-ubuntu22.04

# export timezone - for python3.8
ENV TZ=Europe/Tallinn

# place timezone data /etc/timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# python
RUN apt update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt-get -y install python3.8 python3.8-dev python3.8-distutils python3.8-venv

# venv
ENV VIRTUAL_ENV=/opt/venv
RUN python3.8 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# psycopg2
RUN apt-get update \
    && apt-get -y install libpq-dev gcc


# cv2 libaries
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
#
# git (for yolo)
RUN apt-get install git -y

COPY requirements.txt /var/www/requirements.txt
RUN pip install -r /var/www/requirements.txt

## Copy project
COPY project /var/www/project
COPY app.py /var/www/app.py
COPY config.py /var/www/config.py
COPY .env /var/www/.env

WORKDIR /var/www

#ENTRYPOINT ["flask", "--app", "app", "run"]
#RUN nvidia-smi
#ENTRYPOINT ["bash"]
ENTRYPOINT ["waitress-serve", "--host", "0.0.0.0", "--call", "project:create_app"]
