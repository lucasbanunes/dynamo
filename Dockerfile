FROM python:3.9.15

# Exposing jupyter port
EXPOSE 8888

#Installing extra dependencies
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD jupyter lab --port 8888 --no-browser --allow-root --ip 0.0.0.0 --NotebookApp.token='' --NotebookApp.password=''