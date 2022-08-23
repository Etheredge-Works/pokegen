FROM python:3.8
COPY demo/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY README.md setup.py /app/
COPY src /app/src
WORKDIR /app
RUN ls 
# RUN pip install -e .
RUN python setup.py install
COPY demo/ /app/demo
WORKDIR /app/demo
CMD ["streamlit", "run", "demo.py"]