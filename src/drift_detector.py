import pandas as pd
from pyspark.sql import SparkSession
# from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, FloatType
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import mlflow.spark
from pyspark.ml import PipelineModel


spark = SparkSession.builder \
    .appName("Data Drift Detector App") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores","8") \
    .getOrCreate()
    
def detect_raw_drift(reference_data_path, new_data_path):
    try:
        # preprocessing_pipeline = PipelineModel.load("models/preprocessing_pipeline")
        # referenec_spark_df = spark.read.parquet("data/processed/train_features.parquet")
        # # current_spark_df = spark.read.parquet("data/processed/val_features.parquet")
        # current_raw_spark_df = spark.read.csv(new_data_path, header = True, inferSchema=True)
        # current_spark_df = preprocessing_pipeline.transform(current_raw_spark_df)
        # vector_to_array_udf = udf(lambda v: v.toArray().tolist(), ArrayType(FloatType()))
    
        # feature_count = len(referenec_spark_df.select("features").first()['features'])
        # features_names = [f"feature_{i}" for i in range(feature_count)]
        
        # referenec_spark_df = referenec_spark_df.withColumn(
        #     "features_array",
        #     vector_to_array_udf(col("features"))
        # ).select(*[col("features_array")[i].alias(features_names[i]) for i in range(feature_count)])
        
        # current_spark_df = current_spark_df.withColumn(
        #     "features_array",
        #     vector_to_array_udf(col("features"))
        # ).select(*[col("features_array")[i].alias(features_names[i]) for i in range(feature_count) ])
        
        # reference_data = referenec_spark_df.toPandas()
        # current_data = current_spark_df.toPandas()
        
        reference_data = pd.read_csv(reference_data_path)
        current_data = pd.read_csv(new_data_path)
        
        # Select only the features where drift was introduced
        target_features = ['Age', 'Sex', 'Embarked']
        reference_data = reference_data[target_features]
        current_data = current_data[target_features]
        
        
        data_drift_report = Report(metrics=[DataDriftPreset()])
        
        data_drift_report.run(
            current_data = current_data,
            reference_data = reference_data
        )
        
        data_drift_report.save_html("data_drift_report.html")
        
        report_results = data_drift_report.as_dict()
        has_drift = report_results['metrics'][0]['result']['dataset_drift']
        
        if has_drift:
            print("!!!  ALERT: Data Drift DEtected! Initiating Retraining.")
            #Retraining 
            
        else:
            print("Data Distribution is stable. No Drift Required.")
            
        return has_drift

    except Exception as e:
        print(f"An error occured during drift detection: {e}")
        return False
    
    finally:
        spark.stop()
        
if __name__ == "__main__":
    detect_raw_drift("data/raw/train.csv", "data/raw/simulated_drift_data.csv")    
        