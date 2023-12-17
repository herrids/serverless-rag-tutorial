# Use an official Python runtime as a parent image
FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /usr/src/app


# Install any needed packages specified in requirements.txt
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the rest of your application's code
COPY . .

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Define environment variable
#ENV NAME World

# Run app.py when the container launches
CMD ["python", "./app.py"]
