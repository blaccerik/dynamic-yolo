#FROM ubuntu:20.04
## setup env
#RUN apt update -y
#RUN apt install -y software-properties-common
# install Python3+ and pip
#RUN add-apt-repository universe
#RUN apt install -y python3.8 python3-pip


FROM ubuntu:20.04

# setup env
RUN apt update -y
RUN apt install -y software-properties-common

# install Python3+ and pip
RUN add-apt-repository universe
RUN apt install -y python3.8 python3-pip

# install matplotlib
RUN apt install -y libjpeg-dev zlib1g-dev
RUN pip3 install --upgrade pip setuptools wheel

# psycopg2
RUN apt-get update \
    && apt-get -y install libpq-dev gcc

# cv2 libaries
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# git (for yolo)
RUN apt-get install git -y

COPY requirements.txt /var/www/requirements.txt

RUN pip install -r /var/www/requirements.txt

# Copy project
COPY project /var/www/project
COPY app.py /var/www/app.py
COPY config.py /var/www/config.py
COPY .env /var/www/.env

WORKDIR /var/www

#ENTRYPOINT ["flask", "--app", "app", "run"]
ENTRYPOINT ["waitress-serve", "--host", "0.0.0.0", "--call", "project:create_app"]
#RUN pip install -r /var/www/requirements.txt