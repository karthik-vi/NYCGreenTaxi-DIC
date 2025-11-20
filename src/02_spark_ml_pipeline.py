from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline

# 1. Feature Handling (Adapting your Phase 2 features for Spark)
# Spark needs categorical cols (PULocationID) indexed, unlike pandas which often treats them as ints
indexer_pu = StringIndexer(inputCol="PULocationID", outputCol="PULocationID_Idx", handleInvalid="skip")
indexer_do = StringIndexer(inputCol="DOLocationID", outputCol="DOLocationID_Idx", handleInvalid="skip")

# 2. Vector Assembler (The distinct Spark step)
# We combine Phase 2 features + Phase 3 Weather features
assembler = VectorAssembler(
    inputCols=[
        "trip_distance", 
        "passenger_count", 
        "PULocationID_Idx", 
        "DOLocationID_Idx",
        "PRCP",  # <--- NEW: Precipitation from Weather Data
        "TAVG"   # <--- NEW: Avg Temp from Weather Data
    ],
    outputCol="features",
    handleInvalid="skip" # Skip rows with nulls
)

# 3. Model Definition (Random Forest)
rf = RandomForestRegressor(featuresCol="features", labelCol="fare_amount", numTrees=20)

# 4. Pipeline Construction
pipeline = Pipeline(stages=[indexer_pu, indexer_do, assembler, rf])

print("✅ ML ENGINEERING SUCCESS: Spark Pipeline defined with Weather features.")
# print(pipeline.fit(train_df)) # Uncomment if data is loaded