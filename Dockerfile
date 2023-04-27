FROM nvidia/cuda:12.0.0-devel-ubuntu20.04
USER root
RUN apt-get update && apt-get install -y python3 python3-pip sudo
RUN apt-get -y update
RUN apt-get -y install git

WORKDIR /app
COPY . /app

RUN pip install -r requirement.txt
EXPOSE 8000
CMD ["uvicorn","serve:app","--port","8000","--host","0.0.0.0","--reload"]
#CMD ["python3", "main.py"] 