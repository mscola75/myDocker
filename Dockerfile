FROM python
MAINTAINER mscola@gmail.com
RUN git clone -q  https://github.com/mscola75/myDocker.git
WORKDIR myDocker
cmd ["python", "sample.py"]

