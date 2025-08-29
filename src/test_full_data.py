import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import mlflow.spark
import os
from pyspark.ml.classification import GBTClassificationModel, RandomForestClassificationModel, DecisionTreeClassificationModel, LogisticRegressionModel

spark = SparkSession.builder \
    .appName("TItanic Batch Prediction") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "8") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .getOrCreate()
    
spark.sparkContext.setLogLevel("ERROR")    

mlflow.set_tracking_uri("http://127.0.0.1:5000")

model_path = "models/Classifier"



test_df = spark.read.csv("data/raw/test.csv", header = True, inferSchema = True)

preprocessing_pipeline = PipelineModel.load("models/preprocessing_pipeline")

try:
        loaded_model = GBTClassificationModel.load(model_path)
except Exception:
    try:
        loaded_model = RandomForestClassificationModel.load(model_path)
    except Exception:
        try:
            loaded_model = DecisionTreeClassificationModel.load(model_path)
        except Exception:
            loaded_model = LogisticRegressionModel.load(model_path)


processed_test_df = preprocessing_pipeline.transform(test_df)

predictions = loaded_model.transform(processed_test_df)

submission_df = predictions.select("PassengerId","prediction")
submission_df = submission_df.withColumnRenamed("prediction", "Survived")
submission_df = submission_df.withColumn('Survived', submission_df['Survived'].cast('integer'))

pandas_submission_df = submission_df.toPandas()

pandas_submission_df.to_csv("submission.csv", index=False)

spark.stop()

print("Prediction file 'submission.csv' created successfully.")