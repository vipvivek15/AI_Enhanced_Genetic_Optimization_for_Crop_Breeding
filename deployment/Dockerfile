# Dockerfile

# Use Python base image
FROM python:3.10

# Set workdir inside container
WORKDIR /app

# Copy all files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Run the app
CMD ["python", "app/app.py"]

