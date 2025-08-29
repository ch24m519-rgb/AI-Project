import mlflow.spark
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import os
from mlflow.tracking import MlflowClient
from pyspark.ml import PipelineModel
from pyspark.ml.classification import GBTClassificationModel, RandomForestClassificationModel, DecisionTreeClassificationModel, LogisticRegressionModel
import uvicorn

spark = SparkSession.builder \
    .appName("Titanic Model Deployment") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "8") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .getOrCreate()


mlflow.set_tracking_uri("http://127.0.0.1:5000")

model_path = "models/Classifier"

client = MlflowClient()


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




#Load the saved preprocessing pipeline
preprocessing_pipiline = PipelineModel.load("models/preprocessing_pipeline")

def preprocess_data(data):
    spark_df = spark.createDataFrame([data])
    processed_data = preprocessing_pipiline.transform(spark_df)
    return processed_data


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from typing import Optional

app = FastAPI()

# Define Pydantic model for input validation
class Passenger(BaseModel):
    PassengerId: int
    Pclass: int
    Name: Optional[str] = None
    Sex: Optional[str] = None
    Age: Optional[float] = None
    SibSp: Optional[int] = None
    Parch: Optional[int] = None
    Ticket: Optional[str] = None
    Fare: Optional[float] = None
    Cabin: Optional[str] = None
    Embarked: Optional[str] = None

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

@app.post("/predict")
async def predict(passenger: Passenger):
    try:
        # Convert Pydantic model to dict and create Spark DataFrame
        raw_data = passenger.dict()
        raw_df = spark.createDataFrame([raw_data], schema=input_schema)

        # Preprocess and predict
        processed_df = preprocessing_pipiline.transform(raw_df)
        predictions = loaded_model.transform(processed_df)
        result = predictions.select("prediction").first()["prediction"]
        
        status = "Survived" if result == 1 else "Not-Survived"

        return {
            "prediction": int(result),
            "Survived" : status        
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))




if __name__ == "__main__":
    uvicorn.run("eval:app", host="0.0.0.0", port=5050, reload=True)


