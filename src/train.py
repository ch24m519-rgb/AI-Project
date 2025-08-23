import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.ml.linalg import DenseVector
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from mlflow.models.signature import infer_signature


spark = SparkSession.builder \
    .appName("Titanic Dataset Training") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "8") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .getOrCreate()

train_df = spark.read.parquet("data/processed/train_features.parquet", header=True, inferSchema=True)
# train_with_features = pip
val_df = spark.read.parquet("data/processed/val_features.parquet", header=True, inferSchema=True)

print("tEST fEATURES ",val_df.columns)


lr = LogisticRegression(featuresCol = "features",labelCol="Survived")

paramGrid = ParamGridBuilder() \
    .addGrid(lr.maxIter, [10,50,100]) \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()
    
binaryEvaluator = BinaryClassificationEvaluator(
    labelCol = "Survived",
    rawPredictionCol = "rawPrediction",
    metricName = "areaUnderROC"
)

evaluator = MulticlassClassificationEvaluator(
    labelCol = "Survived",
    predictionCol = "prediction"    
)

cv = CrossValidator(
    estimator = lr,
    estimatorParamMaps = paramGrid,
    evaluator= binaryEvaluator,
    numFolds = 5,
    parallelism = 4
)

mlflow.set_experiment("Titanic-Project")

model_name ="TitanicLogisticRegressionModel"
with mlflow.start_run(run_name="LogisticRegression-CVrun"):
    

    cvModel = cv.fit(train_df)

    bestModel = cvModel.bestModel
    
    params = bestModel.extractParamMap()
    for p,v in params.items():
        mlflow.log_param(str(p.name), v)
    
    print("Best Params: ", bestModel._java_obj.parent().extractParamMap())

    predictions = bestModel.transform(val_df)
    auc = binaryEvaluator.evaluate(predictions)
    mlflow.log_metric("Test_AUC", auc)
    
    accuracy_val = evaluator.setMetricName("accuracy").evaluate(predictions)
    precision_val = evaluator.setMetricName("precisionByLabel").evaluate(predictions)
    recall_val = evaluator.setMetricName("recallByLabel").evaluate(predictions)
    f1_val = evaluator.setMetricName("f1").evaluate(predictions)
    
    print("Accuracy: ",accuracy_val)
    print("Precision: ", precision_val)
    print("Recall: ",recall_val)
    print("F1-score: ", f1_val)
    
    mlflow.log_metric("Accuracy",accuracy_val)
    mlflow.log_metric("Precision",precision_val)
    mlflow.log_metric("Recall",recall_val)
    mlflow.log_metric("F1-score",f1_val)
    
    #saving confusion_matrix and log artifact
    y_true = [int(row['Survived']) for row in predictions.select("Survived").collect()]
    y_pred = [int(row['prediction']) for row in predictions.select('prediction').collect()]
    
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    #saving feature importance and save artifacts
    coef = bestModel.coefficients.toArray()
    sample_features = val_df.select("features").first()["features"]
    feature_count = len(sample_features)
    feature_names = [f"feature_{i}" for i in range(feature_count)]
    coef_df = pd.DataFrame({'Feature': feature_names, 'coefficient': coef})
    
    coef_df_sorted = coef_df.reindex(coef_df.coefficient.abs().sort_values(ascending=False).index)
    
    top_n = 10
    coef_df_top = coef_df_sorted.head(top_n)
    plt.figure(figsize=(12,6))
    sns.barplot(x='coefficient', y='Feature', data=coef_df_top)
    plt.title('Logistic Regression Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    mlflow.log_artifact('feature_importance.png')


    print("BEST PARAMS: ", {p.name: v for p,v in params.items()})
    print("Test AUC = ", auc)
        
    #best model
    mlflow.spark.log_model(bestModel, artifact_path = "model", registered_model_name=model_name)
    
    sample_features = val_df.select("features").first()["features"]
    input_example = pd.DataFrame([sample_features.toArray()], 
                                columns=[f"feature_{i}" for i in range(len(sample_features))])
        
    
    #register the model in the registry
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    registered_model = mlflow.register_model(model_uri, model_name)
    print(f"Registered Model Version: {registered_model.version}")

    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=registered_model.version,
        stage = "Staging", # or can be set to "Production"
        archive_existing_versions = True
    )
    client.set_model_version_tag(name=model_name,
                             version=registered_model.version,
                             key="AUC",
                             value=str(auc))

    prod_model = client.get_latest_versions(name=model_name, stages=["Production"])
    if prod_model:
        prod_run_id = prod_model[0].run_id
        prod_metrics = client.get_run(prod_run_id).data.metrics
        current_prod_auc = prod_metrics.get("Test_AUC",0)
        # current_prod_version = prod_model[0].version
        # prod_model_uri = f"models:/{model_name}/Production"
        # #load production model
        # current_prod_model = mlflow.spark.load_model(prod_model_uri)
        # prod_predictions = current_prod_model.transform(test_df)
        # current_prod_auc = binaryEvaluator.evaluate(prod_predictions)
        
        if auc > current_prod_auc:
            print(f"New model AUC {auc:.4f} > Production AUC {current_prod_auc:.4f} So promoting to Production")
            client.transition_model_version_stage(
                name=model_name,
                version = registered_model.version,
                stage="Production",
                archive_existing_versions=True
            )
            
        else:
            print(f"New model AUC {auc:.4f} <= Production AUC {current_prod_auc:.4f} So keeping in Staging")
    else:
        #no production model exist, promote directly
        print("No Production model found So promoting new model to Production")
        client.transition_model_version_stage(
            name = model_name,
            version = registered_model.version,
            stage = "Production",
        )
        
        
spark.stop()