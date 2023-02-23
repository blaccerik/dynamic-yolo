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

#RUN pip3 install matplotlib

#RUN apk --update add bash nano

# numpy
# RUN apk --no-cache add musl-dev linux-headers g++

# matplot lib depen

# RUN pip install numpy==1.23.4
# RUN pip install matplotlib
#ENV STATIC_URL /static
#ENV STATIC_PATH /var/www/app/static
COPY requirements.txt /var/www/requirements.txt
ENTRYPOINT [ "bash"]
#RUN pip install -r /var/www/requirements.txt