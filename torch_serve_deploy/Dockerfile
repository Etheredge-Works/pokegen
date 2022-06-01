FROM python:3.6
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install 'uvicorn[standard]' fastapi
CMD uvicorn deploy.app:app -l 0.0.0.0:8000