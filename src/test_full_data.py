import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import mlflow.spark

spark = SparkSession.builder \
    .appName("TItanic Batch Prediction") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "8") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .getOrCreate()
    
spark.sparkContext.setLogLevel("ERROR")    

test_df = spark.read.csv("data/raw/test.csv", header = True, inferSchema = True)

preprocessing_pipeline = PipelineModel.load("models/preprocessing_pipeline")

model_name = "TitanicLogisticRegressionModel"
model_uri = f"models:/{model_name}/Production"

loaded_model = mlflow.spark.load_model(model_uri)

processed_test_df = preprocessing_pipeline.transform(test_df)

predictions = loaded_model.transform(processed_test_df)

submission_df = predictions.select("PassengerId","prediction")
submission_df = submission_df.withColumnRenamed("prediction", "Survived")
submission_df = submission_df.withColumn('Survived', submission_df['Survived'].cast('integer'))

pandas_submission_df = submission_df.toPandas()

pandas_submission_df.to_csv("submission.csv", index=False)

spark.stop()

print("Prediction file 'submission.csv' created successfully.")