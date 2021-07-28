FROM python:3.6
RUN pip install streamlit
COPY *.py /app/
WORKDIR /app
CMD ["streamlit", "run", "demo.py"]