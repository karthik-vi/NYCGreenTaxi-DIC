import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, to_date, to_timestamp, when, round
from pyspark.sql.types import DoubleType

def run_weather_analytics():
    # 1. Setup
    spark = SparkSession.builder.appName("Phase3_PartB_Weather_Analytics").getOrCreate()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "..", "data")
    
    # Inputs
    taxi_path = os.path.join(data_dir, "part_a_cleaned_taxi.csv")
    weather_path = os.path.join(data_dir, "NYC-Weather-NOAA-2023.csv")
    final_output = os.path.join(data_dir, "integrated_data.csv")

    print(f"[PART B] Loading Data...")
    taxi_df = spark.read.csv(taxi_path, header=True, inferSchema=True)
    weather_df = spark.read.csv(weather_path, header=True, inferSchema=True)

    # 2. Join Datasets
    # ---------------------------------------------------------
    # Ensure matching types for join keys
    taxi_df = taxi_df.withColumn("join_date", to_date(to_timestamp(col("parsed_datetime"), "yyyy-MM-dd")))
    
    weather_subset = weather_df.select(
        col("DATE").alias("weather_date"),
        col("TAVG").cast(DoubleType()), 
        col("PRCP").cast(DoubleType()), 
        col("SNOW").cast(DoubleType())
    ).withColumn("weather_date", to_date(col("weather_date"))).na.fill(0.0)

    print("[PART B] Joining Taxi & Weather Data...")
    combined_df = taxi_df.join(weather_subset, taxi_df.join_date == weather_subset.weather_date, "inner")

    # 3. Generate Insights (Requirement: 3+ Insights)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("PART B: MULTI-SOURCE ANALYTICS INSIGHTS")
    print("="*50)

    # Insight 1
    print("\n[Insight 1] Impact of Rain on Trip Behavior:")
    combined_df.withColumn("Condition", when(col("PRCP") > 0.1, "Rainy").otherwise("Clear")) \
               .groupBy("Condition") \
               .agg(
                   count("*").alias("Total_Trips"),
                   round(avg("trip_distance"), 2).alias("Avg_Distance"),
                   round(avg("total_amount"), 2).alias("Avg_Fare")
               ).show()

    # Insight 2
    print("\n[Insight 2] Temperature vs. Taxi Demand:")
    combined_df.withColumn("Temp_Bin", (col("TAVG") / 10).cast("int") * 10) \
               .groupBy("Temp_Bin") \
               .agg(count("*").alias("Trip_Count"), round(avg("total_amount"), 2).alias("Avg_Fare")) \
               .orderBy("Temp_Bin") \
               .show()

    # Insight 3
    print("\n[Insight 3] Snow Impact on Fares:")
    combined_df.filter(col("SNOW") > 0) \
               .select(avg("total_amount").alias("Avg_Fare_Snowy_Day"), count("*").alias("Snow_Trips")) \
               .show()

    # 4. Save Integrated Data for Modeling
    # ---------------------------------------------------------
    print(f"Saving fully integrated dataset to {final_output}...")
    
    # Force Cleanup of previous file/folder
    if os.path.exists(final_output):
        try:
            if os.path.isdir(final_output): shutil.rmtree(final_output)
            else: os.remove(final_output)
        except PermissionError:
            print("Error: Could not delete old file. Ensure it is not open.")
            return

    # FIX: Removed 'PU_Borough_Idx' which caused the error
    # Kept 'PU_Zone', 'PULocationID', 'DOLocationID' for the model to use later
    keep_cols = ["trip_distance", "passenger_count", "hour", "day_of_week", 
                 "TAVG", "PRCP", "SNOW", "total_amount", 
                 "PULocationID", "DOLocationID", "PU_Zone"]
    
    # Save via Pandas
    combined_df.select(keep_cols).toPandas().to_csv(final_output, index=False)
    print("Save Complete.")

if __name__ == "__main__":
    run_weather_analytics()