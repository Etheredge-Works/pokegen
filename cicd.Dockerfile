FROM dvcorg/cml-py3
COPY requirements.txt /tmp/pip-tmp/
RUN apt-get update && apt-get install -y default-libmysqlclient-dev \
   && rm -rf /var/lib/apt/lists/* \
   && pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
   && rm -rf /tmp/pip-tmp