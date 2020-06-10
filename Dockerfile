FROM python
MAINTAINER mscola@gmail.com
RUN pip install numpy
RUN pip install pandas
RUN pip install mysql-connector
RUN pip install mysql-connector-python
RUN pip install scikit-learn
RUN pip install scipy

WORKDIR /app
cmd ["python", "B2C_collaborative.py"]
