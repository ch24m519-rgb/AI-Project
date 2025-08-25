# Use a Python base image. A version with a pre-configured environment is ideal.
FROM python:3.10-slim

# Set environment variables for Spark and MLflow.
# These will configure the Spark application inside the container.
ENV SPARK_HOME=/opt/spark
ENV PATH="${PATH}:${SPARK_HOME}/bin"

# Install system dependencies, including Java for Spark
RUN apt-get update && apt-get install -y default-jre  wget

# Download and install Spark
RUN wget https://archive.apache.org/dist/spark/spark-3.3.0/spark-3.3.0-bin-hadoop3.tgz -O /tmp/spark.tgz && \
    tar -xzf /tmp/spark.tgz -C /opt && \
    ln -s /opt/spark-3.3.0-bin-hadoop3 ${SPARK_HOME} && \
    rm /tmp/spark.tgz

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into the container
# This includes src/, requirements.txt, and the models/ directory
COPY requirements.txt .
COPY src/ src/
COPY models/ models/
COPY data/raw/ data/raw/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the Flask app will run on
EXPOSE 5000

# Set the command to run the application
# This is the entry point for your API
CMD ["python", "src/eval.py"]
