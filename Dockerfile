FROM python
MAINTAINER mscola@gmail.com
RUN git clone -q  https://github.com/mscola75/myDocker.git
RUN pip install numpy
RUN pip install pandas
RUN pip install mysql-connector
RUN pip install mysql-connector-python
RUN pip install scikit-learn
RUN pip install scipy

WORKDIR myDocker
cmd ["python", "B2C_collaborative.py"]
