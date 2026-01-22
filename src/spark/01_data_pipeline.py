import os
import shutil
import time
import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, count, avg, desc, hour, when, to_timestamp, to_date, dayofweek

def run_data_pipeline():
    # 1. Initialize Spark
    spark = SparkSession.builder \
        .appName("Phase3_PartA_Taxi_Pipeline") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    # --- Paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "..", "data")
    
    taxi_raw_path = os.path.join(data_dir, "2023_Green_Taxi_Trip_Data_20250926.csv")
    zone_path = os.path.join(data_dir, "taxi_zone_lookup.csv")
    
    # Outputs
    part_a_output = os.path.join(data_dir, "part_a_cleaned_taxi.csv")
    analytics_out = os.path.join(data_dir, "part_a_analytics.csv")

    # 2. Ingestion
    print("\n[PART A] Loading Taxi Data...")
    taxi_df = spark.read.csv(taxi_raw_path, header=True, inferSchema=True)
    zones_df = spark.read.csv(zone_path, header=True, inferSchema=True)

    # 3. Transformations (5+ Required)
    # ---------------------------------------------------------
    print("[PART A] Applying 5 Transformations... (Casting, Joining, Filtering, Binning, Aggregating)")
    
    # T1 & 2: Type Casting & Date Extraction - Joining
    taxi_df = taxi_df.withColumn("parsed_datetime", to_timestamp(col("lpep_pickup_datetime"), "MM/dd/yyyy hh:mm:ss a")) \
                     .withColumn("join_date", to_date(col("parsed_datetime"))) \
                     .withColumn("hour", hour(col("parsed_datetime"))) \
                     .withColumn("day_of_week", dayofweek(col("parsed_datetime")))

    # T3: Filtering (Valid trips)
    taxi_df = taxi_df.filter((col("fare_amount") > 0) & (col("trip_distance") > 0))

    # T4: Feature Engineering (Rush Hour Binning)
    taxi_df = taxi_df.withColumn("Time_Bin", 
        when((col("hour") >= 16) & (col("hour") < 20), "Rush_Hour")
        .otherwise("Off_Peak")
    )
    
    # 4. Advanced Analytics (10 pts)
    # ---------------------------------------------------------
    # Feature 1: Complex SQL Join
    # Joining the Lookup table TWICE (once for Pickup, once for Dropoff)
    print("[PART A] Advanced Feature 1: Complex Joins (Zones)...")
    
    pu_zones = zones_df.select(col("LocationID").alias("PU_ID"), col("Zone").alias("PU_Zone"), col("Borough").alias("PU_Borough"))
    do_zones = zones_df.select(col("LocationID").alias("DO_ID"), col("Zone").alias("DO_Zone"), col("Borough").alias("DO_Borough"))
    
    df = taxi_df.join(pu_zones, taxi_df.PULocationID == pu_zones.PU_ID, "left") \
                .join(do_zones, taxi_df.DOLocationID == do_zones.DO_ID, "left")

    # Feature 2: Window Functions
    # Calculate a "Fare Volatility" metric: Difference between current fare and avg fare for that Borough
    print("[PART A] Advanced Feature 2: Window Functions...")
    windowSpec = Window.partitionBy("PU_Borough")
    df = df.withColumn("Borough_Avg_Fare", avg("total_amount").over(windowSpec)) \
           .withColumn("Fare_Deviation", col("total_amount") - col("Borough_Avg_Fare"))

    # 5. Performance Comparison (15 pts)
    # ---------------------------------------------------------
    print("\n[PART A] Performance Benchmark: Aggregation")
    
    # T5: Aggregating
    t0 = time.time()
    spark_res = df.groupBy("PU_Borough").agg(avg("total_amount")).collect()
    spark_time = time.time() - t0
    
    t0 = time.time()
    pdf_raw = pd.read_csv(taxi_raw_path)
    pandas_res = pdf_raw.groupby("PULocationID")['total_amount'].mean()
    pandas_time = time.time() - t0
    
    print(f"   >>> Spark Time: {spark_time:.4f}s | Pandas Time: {pandas_time:.4f}s")

    # 6. Save Data for Model Training (Cleaning up previous runs)
    print("Saving Cleaned Taxi Data...")
    
    # Cleanup Function
    def clean_path(path):
        if os.path.exists(path):
            try:
                if os.path.isdir(path): shutil.rmtree(path)
                else: os.remove(path)
            except PermissionError:
                print(f"   Error cleaning {path}")

    clean_path(part_a_output)
    clean_path(analytics_out)

    # Save Main Data
    target_cols = ["trip_distance", "passenger_count", "hour", "day_of_week", "total_amount", "PU_Zone", "PU_Borough_Idx"]
    
    # Note: We keep PU_Zone for indexing in next step
    # Saving using Pandas to avoid Windows Hadoop errors
    df.toPandas().to_csv(part_a_output, index=False)
    print(f"Saved: {part_a_output}")

    # Save Analytics Summary (Aggregating)
    summary = df.groupBy("PU_Borough").agg(count("*").alias("trips"), avg("total_amount").alias("avg_fare"))
    summary.toPandas().to_csv(analytics_out, index=False)
    print(f"Saved: {analytics_out}")

if __name__ == "__main__":
    run_data_pipeline()