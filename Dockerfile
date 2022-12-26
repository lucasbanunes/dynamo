FROM python:3.9.15

ARG default_jupyter_port=8888
ENV jupyter_port=${default_jupyter_port}

#Installing extra dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt
WORKDIR /workspace
CMD jupyter lab --port ${jupyter_port} --no-browser --allow-root --ip 0.0.0.0 --NotebookApp.token='' --NotebookApp.password=''