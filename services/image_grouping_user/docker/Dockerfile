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

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy the requirements file first to take advantage of Docker's build cache
COPY docker/requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /app

# Make sourceless .pyc modules importable: copy from __pycache__ to package dirs
RUN set -eux; \
    py_tag="cpython-38"; \
    find /app -type f -path "*/__pycache__/*.${py_tag}.pyc" | while read -r f; do \
      pkg_dir="$(dirname "$(dirname "$f")")"; \
      base="$(basename "$f")"; \
      mod="${base%%.${py_tag}.pyc}"; \
      cp -v "$f" "$pkg_dir/$mod.pyc"; \
    done


#COPY __pycache__/ /app/__pycache__/
#COPY person_recognition/__pycache__/ /app/person_recognition/__pycache__/
#COPY image_face_detection/__pycache__/ /app/image_face_detection/__pycache__/

#COPY cache/ /app/cache/
#COPY individuals_portraits/ /app/individuals_portraits/

COPY docker/docker-entrypoint.sh /usr/local/bin/
COPY docker/healthcheck.py /app/docker/healthcheck.py

RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p person_recognition/logs image_face_detection/logs


# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
 CMD ["python", "docker/healthcheck.py"]

# Entrypoint
ENTRYPOINT ["tini", "--", "/usr/local/bin/docker-entrypoint.sh"]