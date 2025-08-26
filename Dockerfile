# --- Stage 1: Base Image ---
FROM python:3.12-slim-bullseye

# --- Environment Variables ---
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# --- Set Working Directory ---
WORKDIR /app

# --- Install System Dependencies FIRST ---
# This layer will be cached and rarely re-run.
RUN apt-get update && \
    apt-get install -y --no-install-recommends gnupg && \
    rm -rf /var/lib/apt/lists/*

# --- Install Python Dependencies ---
# This layer only re-runs if requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Install Playwright Browsers ---
# This is the slowest step, so we run it before copying our app code.
# This layer will be cached and will only re-run if requirements.txt changes.
RUN playwright install --with-deps

# --- Copy Application Code LAST ---
# Now, we copy our application code. Only changes to these files
# will trigger a rebuild from this point forward.
COPY . .

# --- Expose Port ---
EXPOSE 8000

# --- Start Command ---
CMD ["uvicorn", "index:py", "--host", "0.0.0.0", "--port", "$PORT"]