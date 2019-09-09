# use python 3.7 runtime as parent image
FROM python:3.7.4-alpine3.9

# Set the working directory to git repo root
WORKDIR ../.

# Copy the current directory contents into the container at root
COPY . ../.

# Install any needed packages specified in requirements.txt
# note that not all packages will be required
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

# Define custom env vars as you need
ENV APPNAME jira-label-weak-supervision
ENV AUTHOR shan-96

# Run python when the container launches
CMD ["python3"]