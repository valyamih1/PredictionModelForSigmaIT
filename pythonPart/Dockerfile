FROM ubuntu:22.10

RUN apt-get update && apt-get install -y python3.10 python3-pip

COPY . /opt/.

RUN pip3 install -r /opt/requirements.txt

CMD python3 /opt/main.py