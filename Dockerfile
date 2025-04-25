# Use official Python image as base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app_finaal

# Copy requirements.txt to the container and install dependencies
COPY requirements.txt /app_finaal/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application into the container
COPY . /app_finaal

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run FastAPI using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
