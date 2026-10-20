FROM python:3.10-bullseye

# Install Java 11 (required by PySpark)
RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    curl \
    wget \
    procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyG dependencies
RUN pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

COPY . .