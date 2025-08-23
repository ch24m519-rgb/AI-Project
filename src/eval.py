import mlflow.spark
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

spark = SparkSession.builder \
    .appName("Titanic Model Deployment") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "8") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .getOrCreate()

model_name = "TitanicLogisticRegressionModel"
model_uri = f"models:/{model_name}/Production"
loaded_model = mlflow.spark.load_model(model_uri)


#Load the saved preprocessing pipeline
preprocessing_pipiline = PipelineModel.load("models/preprocessing_pipeline")

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
    app.run(host="0.0.0.0", port = 5000)


