FROM eclipse-temurin:11-jre-jammy

# Set environment variables for Spark and MLflow.
# These will configure the Spark application inside the container.
ENV SPARK_HOME=/opt/spark
ENV PATH="${PATH}:${SPARK_HOME}/bin"
ENV DOCKER_ENV=1
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Set the working directory inside the container
WORKDIR /app


# Install system dependencies, including Java for Spark
RUN apt-get update && \
    apt-get install -y wget python3 python3-pip python3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Download and install Spark
RUN wget https://archive.apache.org/dist/spark/spark-3.3.0/spark-3.3.0-bin-hadoop3.tgz -O /tmp/spark.tgz && \
    tar -xzf /tmp/spark.tgz -C /opt && \
    ln -s /opt/spark-3.3.0-bin-hadoop3 ${SPARK_HOME} && \
    rm /tmp/spark.tgz


# Install Python dependencies (copy requirements file)
COPY requirements.txt .
RUN pip3 install --default-timeout=120 --no-cache-dir -r requirements.txt

# Copy all project files into the container
# This includes src/and the models/ directory

COPY src/ src/
COPY models/ models/

COPY data/raw/ data/raw/
COPY data/processed/ data/processed/


# Expose the port the Flask app will run on
EXPOSE 5000

# Set the command to run the application
# This is the entry point for your API

CMD ["python3"]     
#, "src/titanic_preprocess.py"]
# CMD ["python3", "src/train.py"]
# CMD ["python3", "src/eval.py"]
