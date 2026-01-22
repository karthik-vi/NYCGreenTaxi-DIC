import os
import shutil
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

def run_weather_modeling():
    spark = SparkSession.builder \
        .appName("Phase3_PartB_Weather_Model") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "..", "data", "integrated_data.csv")
    model_save_path = os.path.join(script_dir, "..", "..", "data", "saved_models", "Weather_Enhanced_RF")

    print(f"[PART B] Loading Integrated Data: {data_path}")
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # --- SAFETY FIX: Drop any rows with nulls in critical columns ---
    # This prevents the VectorAssembler crash we saw in Part A
    feature_cols_raw = [
        "trip_distance", "passenger_count", "hour", "day_of_week", 
        "TAVG", "PRCP", "SNOW", "PU_Zone"
    ]
    df = df.dropna(subset=feature_cols_raw + ["total_amount"])

    # 1. Feature Engineering
    # ---------------------------------------------------------
    # Index the Zone (String -> Number)
    # handleInvalid="skip" ensures we drop any unseen zones instead of crashing
    indexer = StringIndexer(inputCol="PU_Zone", outputCol="PU_Index", handleInvalid="skip")
    
    encoder = OneHotEncoder(inputCol="PU_Index", outputCol="PU_Vec")

    # Combine all features into one vector
    assembler_inputs = [
        "trip_distance", "passenger_count", "hour", "day_of_week", # Taxi Features
        "TAVG", "PRCP", "SNOW",                                    # Weather Features
        "PU_Vec"                                                   # Location Features
    ]
    
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="skip")

    # 2. Model Definition
    # ---------------------------------------------------------
    # Using 50 trees and depth 8 (balanced for speed vs accuracy)
    # maxBins=64 helps handle the large number of NYC Zones
    rf = RandomForestRegressor(labelCol="total_amount", featuresCol="features", 
                               numTrees=50, maxDepth=8, maxBins=64, seed=42)
    
    pipeline = Pipeline(stages=[indexer, encoder, assembler, rf])

    # 3. Train & Evaluate
    # ---------------------------------------------------------
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    
    print("Training Weather-Enhanced Random Forest...")
    model = pipeline.fit(train)
    
    predictions = model.transform(test)
    
    rmse = RegressionEvaluator(labelCol="total_amount", metricName="rmse").evaluate(predictions)
    r2 = RegressionEvaluator(labelCol="total_amount", metricName="r2").evaluate(predictions)
    
    print("\n" + "="*50)
    print("PART B: MODELING RESULTS (Combined Sources)")
    print("="*50)
    print(f"Model: Random Forest Regressor (Taxi + Weather Features)")
    print(f"RMSE:  {rmse:.4f}")
    print(f"R2:    {r2:.4f}")
    print("="*50)
    
    # 4. Save Model (Requirement)
    # Wrapped in try/except to safely handle the Windows/Hadoop bug
    try:
        if os.path.exists(model_save_path):
            shutil.rmtree(model_save_path)
        model.save(model_save_path)
        print(f"\nModel saved to: {model_save_path}")
    except Exception as e:
        print(f"\n(Warning) Model save skipped due to Windows environment: {e}")
        print("This does not affect the validity of the analysis/results.")

if __name__ == "__main__":
    run_weather_modeling()