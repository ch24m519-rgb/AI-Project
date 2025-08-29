import mlflow.spark
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import os
from mlflow.tracking import MlflowClient

spark = SparkSession.builder \
    .appName("Titanic Model Deployment") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "8") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .getOrCreate()



# mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:///mnt/d/wsl_trim3_project/project_titanic/mlruns"))
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns"))


client = MlflowClient()

# List all registered models
registered_models = client.search_registered_models()

for model in registered_models:
    model_name = model.name
    
    prod_versions = client.get_latest_versions(name=model_name, stages=["Production"])
    
    if prod_versions:
        for version in prod_versions:
            print(f"Model: {model_name}")
            print(f" Production Version: {version.version}")
            prod_run_id = version.run_id
            print(f" Run ID: {version.run_id}")
            # model_uri = f"runs:/{prod_run_id}/model"
            # loaded_model = mlflow.spark.load_model(model_uri)
            
            
            model_uri = f"models:/{model_name}/Production"
            print("Model URI: ", model_uri)
            loaded_model = mlflow.spark.load_model(model_uri)
            
            
            # local_model_path = f"/app/mlruns/276474462388578523/e987e243a7e644feb0c6d758b47eda63/artifacts/model"
            # loaded_model = mlflow.spark.load_model(f"file://{local_model_path}")
            
            print("-----")
            
    else:
        print(f"Model: {model_name} has no version in Production stage.")

# model_uri = "file:///app/mlruns/276474462388578523/e987e243a7e644feb0c6d758b47eda63/artifacts/model"
# loaded_model = mlflow.spark.load_model(model_uri)




#Load the saved preprocessing pipeline
preprocessing_pipiline = PipelineModel.load("pipeline/preprocessing_pipeline")

def preprocess_data(data):
    spark_df = spark.createDataFrame([data])
    processed_data = preprocessing_pipiline.transform(spark_df)
    return processed_data


"""
creating API with Flask
"""
from flask import Flask, request, jsonify
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

app = Flask(__name__)

input_schema = StructType([
    StructField("PassengerId", IntegerType(), True),
    StructField("Pclass", IntegerType(), True),
    StructField("Name", StringType(), True),
    StructField("Sex", StringType(), True),
    StructField("Age", DoubleType(), True),
    StructField("SibSp", IntegerType(), True),
    StructField("Parch", IntegerType(), True),
    StructField("Ticket", StringType(), True),
    StructField("Fare", DoubleType(), True),
    StructField("Cabin", StringType(), True),
    StructField("Embarked", StringType(), True)
])

#API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        raw_data = request.json
        raw_df = spark.createDataFrame([raw_data], schema=input_schema)
        
        processed_df = preprocessing_pipiline.transform(raw_df)
        
        predictions = loaded_model.transform(processed_df)
        
        result = predictions.select("prediction").first()["prediction"]
        
        return jsonify({"prediction": int(result)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# spark.stop()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 5050)


