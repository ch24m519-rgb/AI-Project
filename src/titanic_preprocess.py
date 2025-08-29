import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.feature import Imputer, VectorAssembler, VectorIndexer, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col, concat_ws

spark = SparkSession.builder \
    .appName("EndTermTitanic") \
    .config("spark.executor.memory","8g") \
    .config("spark.executor.cores","8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")    

print("Message: Spark Session Created")

df = spark.read.csv("data/raw/train.csv", header=True, inferSchema=True)
# test_df = spark.read.csv("data/raw/titanic_test.csv", header=True, inferSchema=True)
# test_survived_df = spark.read.csv("data/raw/gender_submission.csv", header=True, inferSchema=True)

train_df, val_df = df.randomSplit([0.8,0.2], seed = 42)

train_df.show(10)

categorical_columns = [f.name for f in train_df.schema.fields if isinstance(f.dataType, StringType)]
numerical_columns = [f.name for f in train_df.schema.fields if ((isinstance(f.dataType, IntegerType) or isinstance(f.dataType, DoubleType)) & (f.name != "Survived") )]
print("Categorical Cols: ", categorical_columns)
print("NUmerical COls: ",numerical_columns)


imputer_num = Imputer(
    inputCols=numerical_columns,
    outputCols=numerical_columns,
    strategy="mean"
)



indexOutputCols = [x + "Index" for x in categorical_columns]
oheOutputCols = [x + "OHE" for x in categorical_columns]
stringIndexer = StringIndexer(inputCols=categorical_columns, outputCols=indexOutputCols, handleInvalid="keep")
oheEncoder = OneHotEncoder(inputCols=indexOutputCols, outputCols=oheOutputCols, handleInvalid="keep")

numAssember = VectorAssembler(inputCols=numerical_columns, outputCol="num_raw")
scaler = StandardScaler(inputCol="num_raw", outputCol="num_scaled", withMean=True, withStd=True)
finalAssembler = VectorAssembler(inputCols=["num_scaled"] + oheOutputCols, outputCol="features")


pipeline = Pipeline(
    stages = [
        imputer_num,
        stringIndexer, 
        oheEncoder,
        numAssember,
        scaler,
        finalAssembler
    ]
)

pipeline_model = pipeline.fit(train_df)
predictions = pipeline_model.transform(val_df)

processed_train = pipeline_model.transform(train_df) \
    .select("PassengerId", "features", "Survived")

processed_train.write.mode("overwrite").option("header",True).parquet("data/processed/train_features.parquet")

processed_val = pipeline_model.transform(val_df) \
    .select("PassengerId", "features", "Survived")

processed_val.write.mode("overwrite").option("header",True).parquet("data/processed/val_features.parquet")

pipeline_model.write().overwrite().save("pipeline/preprocessing_pipeline")

print("train_features.parquet Saved Successfully")
print("test_features.parquet Saved Successfully")

spark.stop()

