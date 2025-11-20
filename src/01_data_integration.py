from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, col, lit

# Initialize Spark
spark = SparkSession.builder.appName("Phase3_DryRun_Data").getOrCreate()

# 1. Load Your Phase 2 Data (Simulated loading of your actual CSV)
# We use the schema implied by your notebooks: lpep_pickup_datetime, fare_amount, etc.
taxi_df = spark.read.csv("taxi_data.csv", header=True, inferSchema=True)

# 2. Prepare for Join (Convert timestamp to date)
taxi_df = taxi_df.withColumn("date_key", to_date(col("lpep_pickup_datetime")))

# 3. Load/Create Weather Data (The "New" Phase 3 variable)
# In production, read "weather_2023.csv". For dry run, create a dummy dataframe.
weather_data = [("2023-01-01", 0.0, 45), ("2023-01-02", 0.5, 42)] # Date, Precip, Temp
weather_df = spark.createDataFrame(weather_data, ["weather_date", "PRCP", "TAVG"])
weather_df = weather_df.withColumn("weather_date", to_date(col("weather_date")))

# 4. The Integration (Joining the datasets)
enriched_df = taxi_df.join(weather_df, taxi_df.date_key == weather_df.weather_date, "left")

print("✅ DATA ENGINEERING SUCCESS: Integrated Schema created.")
enriched_df.select("lpep_pickup_datetime", "fare_amount", "PRCP", "TAVG").show(5)