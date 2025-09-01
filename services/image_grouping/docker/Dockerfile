FROM python:3.8.20

# Add these lines to your existing Dockerfile
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql17 unixodbc-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
# Install ODBC libraries and OpenCV dependencies
RUN apt-get update && apt-get install -y \
    unixodbc \
    unixodbc-dev \
    libodbc1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install tini only if not installed
RUN if ! command -v tini >/dev/null 2>&1; then \
        apt-get update && apt-get install -y tini && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app/services/image_grouping

# --- copy requirements first for caching ---
COPY docker/requirements.txt ./requirements.txt

# Install deps with cache (only re-run if requirements.txt changes)
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# --- now copy the rest of the code ---
COPY person_recognition/ ./person_recognition/
COPY image_face_detection/ ./image_face_detection/
COPY main.py ./main.py
COPY docker/docker-entrypoint.sh /usr/local/bin/
COPY docker/healthcheck.py ./docker/healthcheck.py
COPY config.py ./config.py
COPY .env ./.env

# Copy logger_config.py files to their respective folders
COPY person_recognition/logger_config.py ./person_recognition/logger_config.py
COPY image_face_detection/logger_config.py ./image_face_detection/logger_config.py

RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p person_recognition/logs image_face_detection/logs

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD ["python", "docker/healthcheck.py"]

# Entrypoint
ENTRYPOINT ["tini", "--", "/usr/local/bin/docker-entrypoint.sh"]
