# Use a slim Python base image
FROM python:3.10-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies. Use --no-cache-dir to keep image size down.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port your FastAPI app will run on
EXPOSE 8000

# Command to run your FastAPI application with Uvicorn
# Use the $PORT environment variable that Render provides
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]