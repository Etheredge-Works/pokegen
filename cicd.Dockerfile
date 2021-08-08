# TODO could be cool to have a build branch
#      that shinks down docker files
FROM dvcorg/cml-py3
COPY requirements.txt /tmp/pip-tmp/
# NOTE can be made smaller, but it makes failure harder to debug
RUN apt-get update
RUN apt-get install -y python3-dev default-libmysqlclient-dev build-essential
RUN rm -rf /var/lib/apt/lists/*
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt
RUN rm -rf /tmp/pip-tmp