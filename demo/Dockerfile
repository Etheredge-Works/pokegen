FROM python:3.8
RUN pip install streamlit
COPY *.py /app/
WORKDIR /app
CMD ["streamlit", "run", "demo.py"]