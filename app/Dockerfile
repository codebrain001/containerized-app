# start by pulling the python image
FROM python:3.8-slim-buster

# Set and create the working directory to /app
WORKDIR /app

# Copy the current directory contents into /app
COPY . /app

# informs docker that this container should listen to network port 5000 at runtime
EXPOSE 5000

# install the dependencies and packages in the requirements file
RUN apt-get -y update
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install -r requirements.txt

#  executing the container
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "wsgi:app"]