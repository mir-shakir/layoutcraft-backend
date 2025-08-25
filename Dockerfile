# --- Stage 1: Base Image ---
# Use a slim, official Python image for a smaller final container size.
# Using a specific version (e.g., 3.12) is better for production than `latest`.
FROM python:3.12-slim

# --- Environment Variables ---
# Set environment variables to prevent Python from writing .pyc files and to buffer output.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# --- Set Working Directory ---
# All subsequent commands will run from this directory inside the container.
WORKDIR /app

# --- Install Python Dependencies ---
# Copy only the requirements file first to take advantage of Docker's layer caching.
# This layer will only be rebuilt if requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Install Playwright Browsers & System Dependencies ---
# This is the most critical step for a server deployment.
# `--with-deps` installs the necessary system libraries (like fonts, graphics libs, etc.)
# that the headless Chromium browser needs to run correctly on a lean Linux server.
# --- Install Playwright Browsers & System Dependencies ---
# First, update the package lists to ensure we can find all dependencies.
RUN apt-get update \
    # Install essential build tools that are missing from the slim image
    && apt-get install -y --no-install-recommends \
    gnupg \
    # Clean up the apt cache to keep the image size small
    && rm -rf /var/lib/apt/lists/*
    && echo "Cache bust: $(date)"
RUN playwright install --with-deps

# --- Copy Application Code ---
# Copy the rest of your application's source code into the container.
COPY . .

# --- Expose Port ---
# Inform Docker that the container listens on port 8000.
# Render will automatically use this, but it's good practice to declare it.
EXPOSE 8000

# --- Start Command ---
# The command to run when the container starts.
# We use "--host 0.0.0.0" to make the server accessible from outside the container.
# Render will automatically set the $PORT environment variable.
# CMD ["uvicorn", "index:app", "--host", "0.0.0.0", "--port", "$PORT"]
CMD uvicorn index:app --host 0.0.0.0 --port $PORT