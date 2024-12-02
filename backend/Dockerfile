# Use the official Python image from Docker Hub
FROM python:3.8.18

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt to the container
COPY requirements.txt .

# Install Python dependencies with longer timeout and no caching
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --default-timeout=100 -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose the port that the Flask app will run on
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP=app.py

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
