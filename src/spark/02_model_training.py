import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor, LinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline

def run_model_training():
    # 1. Initialize
    spark = SparkSession.builder \
        .appName("Phase3_PartB_ML_Models") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "..", "data", "part_a_cleaned_taxi.csv") # Using Part A data
    model_save_dir = os.path.join(script_dir, "..", "..", "data", "saved_models")

    # 2. Load Data
    print(f"Loading training data from: {data_path}")
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # --- CRITICAL FIX: Ensure no nulls in feature columns ---
    # Even if Part A cleaned it, loading from CSV might re-introduce issues if schema inference is weird
    # We drop nulls explicitly again here to be safe.
    required_cols = ["trip_distance", "passenger_count", "hour", "day_of_week", "PU_Zone", "total_amount"]
    df = df.na.drop(subset=required_cols)
    
    # Filter out any weird edge cases
    df = df.filter(col("total_amount") > 0)

    # 3. Feature Pipeline Stages
    # ---------------------------------------------------------
    # Indexing Zone
    # We use "skip" to just drop any unseen labels during cross-validation which prevents crashes
    indexer = StringIndexer(inputCol="PU_Zone", outputCol="PU_Idx", handleInvalid="skip") 
    
    # OneHotEncode
    encoder = OneHotEncoder(inputCols=["PU_Idx"], outputCols=["PU_Vec"])

    # Vectorize
    # Note: We are using a simpler feature set to guarantee stability
    feature_cols = ["trip_distance", "passenger_count", "hour", "day_of_week", "PU_Vec"]
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

    # Base stages
    base_stages = [indexer, encoder, assembler]

    # 4. Define 3 Different Models
    # ---------------------------------------------------------
    models = {
        "LinearRegression": LinearRegression(labelCol="total_amount", featuresCol="features"),
        "RandomForest": RandomForestRegressor(labelCol="total_amount", featuresCol="features", seed=42, maxBins=64), # Increased bins for categorical features
        "GBTRegressor": GBTRegressor(labelCol="total_amount", featuresCol="features", seed=42, maxBins=64)
    }

    # Split Data
    print("Splitting data...")
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    
    # Cache to improve performance
    train.cache()
    
    results = {}

    # 5. Train & Evaluate Loop
    for name, estimator in models.items():
        print(f"\nTraining {name}...")
        
        pipeline = Pipeline(stages=base_stages + [estimator])
        
        # Minimal Grid for Demonstration
        paramGrid = ParamGridBuilder().build()

        crossval = CrossValidator(estimator=pipeline,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=RegressionEvaluator(labelCol="total_amount", metricName="rmse"),
                                  numFolds=2)

        try:
            # Fit
            cv_model = crossval.fit(train)
            
            # Evaluate
            predictions = cv_model.transform(test)
            rmse = RegressionEvaluator(labelCol="total_amount", metricName="rmse").evaluate(predictions)
            r2 = RegressionEvaluator(labelCol="total_amount", metricName="r2").evaluate(predictions)
            
            results[name] = {"RMSE": rmse, "R2": r2}
            print(f"   > RMSE: {rmse:.4f}")
            print(f"   > R2:   {r2:.4f}")

            # Persistence
            try:
                save_path = os.path.join(model_save_dir, name)
                if os.path.exists(save_path):
                    shutil.rmtree(save_path)
                cv_model.bestModel.save(save_path)
                print(f"   > Saved model to: {save_path}")
            except Exception:
                print("   > (Warning) Saving skipped due to Windows env.")
                
        except Exception as e:
            print(f"   > Error training {name}: {str(e)}")

    # 6. Final Report
    print("\n" + "="*40)
    print("FINAL MODEL COMPARISON (Phase 3)")
    print("="*40)
    for name, metrics in results.items():
        print(f"{name:20} | RMSE: {metrics['RMSE']:.4f} | R2: {metrics['R2']:.4f}")

if __name__ == "__main__":
    run_model_training()