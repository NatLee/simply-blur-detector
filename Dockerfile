FROM tensorflow/tensorflow:2.7.4-gpu
WORKDIR /src
COPY ./src /src
COPY ./requirements.txt /src

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

RUN chmod a+x docker-entrypoint.sh
