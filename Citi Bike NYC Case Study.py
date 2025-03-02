# Databricks notebook source
# MAGIC %md
# MAGIC # Citi Bike Trip NYC Case Study
# MAGIC
# MAGIC This analysis examines the last month of bike-sharing data in NYC, sourced from [Citi Bike NYC](https://citibikenyc.com/system-data). The objective is to assess the impact of introducing an insurance coverage fee for trips exceeding 30 minutes and to provide insights into travel distances. The key points of analysis include:
# MAGIC
# MAGIC - Total number of trips covered in the dataset.
# MAGIC - Estimating the potential revenue generated if a fee of $0.20 were charged for each ride exceeding 30 minutes.
# MAGIC - Analyzing trip distances by categorizing them into distance buckets (0-1 km, 2-4 km, 4-9 km, 10+ km) and visualizing the distribution in a diagram.
# MAGIC
# MAGIC ## Notes
# MAGIC - As of March 2, 2025, the most recent available dataset covers January 2025. Therefore, this analysis technically reflects data from the second most recent month rather than the last month.
# MAGIC - Travel distance buckets are assumed to be in kilometers and are defined as follows:
# MAGIC   - 0-1 km: [0, 2)
# MAGIC   - 2-4 km: [2, 4)
# MAGIC   - 4-9 km: [4, 10)
# MAGIC   - 10+ km: [10, ∞)
# MAGIC - The dataset for Jersey City (JC-202501-citibike-tripdata.csv.zip) is not considered for the analysis.
# MAGIC - Charges only apply to rides that exceed 30 minutes (excluding ride durations below or equal to exactly 30 minutes)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data Ingestion, Exploration, and Preparation
# MAGIC
# MAGIC This step involves downloading the dataset from the source, investigate it, and prepare it to answer the questions.
# MAGIC
# MAGIC ### 1.1 Data Ingestion
# MAGIC
# MAGIC - Download the file
# MAGIC - Extract the zipped file
# MAGIC - Load file as a PySpark dataframe

# COMMAND ----------

# DBTITLE 1,Data Ingestion
import urllib.request
import zipfile
import os
import math

from pyspark.sql import functions as F
from pyspark.sql import types as T

# Define the URL and the local file paths
url = "https://s3.amazonaws.com/tripdata/202501-citibike-tripdata.zip"
zip_path = "/tmp/202501-citibike-tripdata.zip"
csv_path = "/tmp/*.csv"

# Download the zip file
urllib.request.urlretrieve(url, zip_path)

# Unzip the file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/tmp")

# Load the CSV as a DataFrame
df_citibikenyc = spark.read.csv("file:" + csv_path, header=True, inferSchema=True)

# Display the DataFrame
display(df_citibikenyc)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Data Profiling
# MAGIC
# MAGIC This part covers some basic data profiling tasks which extracts insights about the ingested data itself.
# MAGIC
# MAGIC - The total number of rows ingested.
# MAGIC - Determining if the inferred schema is correct.
# MAGIC - Statistics about the data.
# MAGIC - Perform rudimentary data quality checks, such as empty or invalid values.

# COMMAND ----------

# DBTITLE 1,Row Count
row_count = df_citibikenyc.count()
print("Total number of rows: " + str(row_count))

# COMMAND ----------

# DBTITLE 1,Print Schema
df_citibikenyc.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC __Conclusion:__ The schema inferred from the CSV files seem to be correct.

# COMMAND ----------

# DBTITLE 1,Summary
display(df_citibikenyc.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC __Conclusion:__ According to the statistics, some end latitudes and end longitudes are 0.0 which cannot be correct. Latitude around 0 means that it is at the same north/south-level as the equator while longitude 0 means that is is at the same west/east-level as Greenwhich. Both attributes would not apply to NYC.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Data Quality Checks
# MAGIC
# MAGIC - Investigate why some end latitudes and longitudes are 0.0 and fix it (if possible).
# MAGIC - Perform rudimentary checks involving started and ended timestamps.

# COMMAND ----------

display(df_citibikenyc[df_citibikenyc.end_lng == 0.0])

# COMMAND ----------

display(df_citibikenyc[df_citibikenyc.end_station_name == 'Bronx WH station'])

# COMMAND ----------

# MAGIC %md
# MAGIC __Conclusion:__ It looks like that trips that ends at _Bronx WH station_ do not provide a valid coordinate. After a Google search, this seems to be the subway station [Whitlock Avenue station](https://en.wikipedia.org/wiki/Whitlock_Avenue_station).
# MAGIC
# MAGIC To fix the coordinates, the coordinates provided by the Wikipedia will be used to replace the 0.0:
# MAGIC - Latitude: 40.827514
# MAGIC - Longitude: -73.886147

# COMMAND ----------

# DBTITLE 1,Update Latitude and Longitude
# Update latitude and longitude
df_citibikenyc = df_citibikenyc.withColumn("end_lat", F.when(df_citibikenyc.end_lat == 0.0, 40.827514).otherwise(df_citibikenyc.end_lat))
df_citibikenyc = df_citibikenyc.withColumn("end_lng", F.when(df_citibikenyc.end_lng == 0.0, -73.886147).otherwise(df_citibikenyc.end_lng))

# COMMAND ----------

# DBTITLE 1,Verify updates
display(df_citibikenyc[df_citibikenyc.end_station_name == 'Bronx WH station'])

# COMMAND ----------

# DBTITLE 1,Investigate start and end timestamps
# Check for null values in started_at and ended_at
display(df_citibikenyc[df_citibikenyc.started_at.isNull()])
display(df_citibikenyc[df_citibikenyc.ended_at.isNull()])

# Check for started_at that is later than ended_at
display(df_citibikenyc[df_citibikenyc.started_at > df_citibikenyc.ended_at])

# COMMAND ----------

# MAGIC %md
# MAGIC __Conclusion:__ The (basic) checks for timestamps seem to be ok.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4 Data Preparation
# MAGIC
# MAGIC Enrich the dataset to support with the analysis tasks.
# MAGIC
# MAGIC - Add a column with ride duration in minutes.
# MAGIC - Add a column with ride distance in km.
# MAGIC - Add a categorical column for distance buckets.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.4.1 Ride Duration in Minutes
# MAGIC
# MAGIC Add a new calculated column for ride duration by subtracting the started_at from ended_at and return the value as minutes.

# COMMAND ----------

df_citibikenyc_with_duration = df_citibikenyc.withColumn("trip_duration_minutes", (F.unix_timestamp(F.col("ended_at")) - F.unix_timestamp(F.col("started_at"))) / 60)
display(df_citibikenyc_with_duration)

# COMMAND ----------

df_citibikenyc_with_duration[df_citibikenyc_with_duration.trip_duration_minutes > 30].count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.4.2 Distance in km
# MAGIC
# MAGIC Since only the start and end coordinates are provided, calculating the exact trip distance is challenging without knowing the precise route taken. To simplify the analysis, the distance will be estimated using the straight-line distance between the start and end coordinates.

# COMMAND ----------

# Source: https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula

# Define the Haversine formula to calculate distance
def haversine_distance(lat1, lon1, lat2, lon2):
    try:
        R = 6371.0  # Radius of the Earth in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance_km = R * c # Distance in km
    # Some rides do not have end coordinates, these will default as -1.0
    except:
        distance_km = -1.0
    return distance_km

# Register the UDF
haversine_udf = F.udf(haversine_distance, T.DoubleType())

# Calculate the distance and add it as a new column
df_citibikenyc_with_distance = df_citibikenyc_with_duration.withColumn(
    "trip_distance_km",
    haversine_udf(
        F.col("start_lat"),
        F.col("start_lng"),
        F.col("end_lat"),
        F.col("end_lng")
    )
)

display(df_citibikenyc_with_distance)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.4.3 Distance Buckets
# MAGIC
# MAGIC An additional categorical attribute for distance buckets, defined with the following labels:
# MAGIC
# MAGIC - [0-2 km]
# MAGIC - [2-4 km]
# MAGIC - [4-10 km]
# MAGIC - [10+ km]
# MAGIC

# COMMAND ----------

df_citibikenyc_final = (df_citibikenyc_with_distance
  .withColumn("distance_category",
              F.when((df_citibikenyc_with_distance.trip_distance_km >= 0.0) & (df_citibikenyc_with_distance.trip_distance_km < 2.0), "0:[0-2 km]")
              .when((df_citibikenyc_with_distance.trip_distance_km >= 2.0) & (df_citibikenyc_with_distance.trip_distance_km < 4.0), "1:[2-4 km]")
              .when((df_citibikenyc_with_distance.trip_distance_km >= 4.0) & (df_citibikenyc_with_distance.trip_distance_km < 10.0), "2:[4-10 km]")
              .when(df_citibikenyc_with_distance.trip_distance_km >= 10.0, "3:[10+ km]")
              .otherwise("4:[N/A]")))

display(df_citibikenyc_final)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Analysis
# MAGIC
# MAGIC With the data prepared, the analysis focuses on the following key questions:
# MAGIC
# MAGIC - How many trips would be covered?
# MAGIC - If your manager thinks we could charge 0.2 USD for each ride that takes longer than 30 minutes, how much revenue could we expect?
# MAGIC - Your manager wants to understand the travel distance in distance buckets (0-1,2-4,4-9,10+).Please make a diagram.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Trips Covered
# MAGIC
# MAGIC > How many trips would be covered?

# COMMAND ----------

# Total trips available in the dataset
total_trips = df_citibikenyc_final.count()

# Total trips over 30 minutes
total_trips_over_30mins = df_citibikenyc_final[df_citibikenyc_final.trip_duration_minutes > 30].count()

print("Total trips: {}".format(total_trips))
print("Trips over 30 minutes: {}".format(total_trips_over_30mins))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Revenue for 30+ Minute Trips

# COMMAND ----------

# Calculate the revenue for trips over 30 minutes by multiplying the number of affected trips with $0.20
revenue_for_trips_over_30mins = total_trips_over_30mins * 0.2

print("Revenue for trips over 30 minutes: ${}".format(revenue_for_trips_over_30mins))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Trips Distribution

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# Convert Spark DataFrame to Pandas DataFrame
df_grouped_pd = df_citibikenyc_final.groupBy("distance_category", "member_casual").count().toPandas()

# Pivot the data for bar chart
df_pivot_pd = df_grouped_pd.pivot(index="distance_category", columns="member_casual", values="count").fillna(0)

# Plot the bar chart using seaborn
df_pivot_pd.reset_index(inplace=True)
df_melted = df_pivot_pd.melt(id_vars="distance_category", var_name="member_casual", value_name="count")

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df_melted, x="distance_category", y="count", hue="member_casual", errorbar=None)

plt.xlabel('Distance Category')
plt.ylabel('Number of Trips')
plt.title('Bar Chart by Distance Category and Member Type')
plt.legend(title='Member Type')

# Add total number of rides on each bar
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()

# COMMAND ----------

# Convert Spark DataFrame to Pandas DataFrame
df_grouped_pd = df_citibikenyc_final[df_citibikenyc_final.trip_duration_minutes > 30].groupBy("distance_category", "member_casual").count().toPandas()

# Pivot the data for bar chart
df_pivot_pd = df_grouped_pd.pivot(index="distance_category", columns="member_casual", values="count").fillna(0)

# Plot the bar chart using seaborn
df_pivot_pd.reset_index(inplace=True)
df_melted = df_pivot_pd.melt(id_vars="distance_category", var_name="member_casual", value_name="count")

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df_melted, x="distance_category", y="count", hue="member_casual", errorbar=None)

plt.xlabel('Distance Category')
plt.ylabel('Number of Trips')
plt.title('Bar Chart by Distance Category and Member Type for Trips Longer than 30 Minutes')
plt.legend(title='Member Type')

# Add total number of rides on each bar
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()

# COMMAND ----------

# Calculate mean trip duration minutes grouped by distance category and member type
df_mean_duration = df_citibikenyc_final.groupBy("distance_category", "member_casual").agg({"trip_duration_minutes": "mean"})

# Convert Spark DataFrame to Pandas DataFrame
df_mean_duration_pd = df_mean_duration.toPandas()

# Pivot the data for bar chart
df_pivot_mean_duration_pd = df_mean_duration_pd.pivot(index="distance_category", columns="member_casual", values="avg(trip_duration_minutes)").fillna(0)

# Plot the bar chart using seaborn
df_pivot_mean_duration_pd.reset_index(inplace=True)
df_melted_mean_duration = df_pivot_mean_duration_pd.melt(id_vars="distance_category", var_name="member_casual", value_name="mean_trip_duration_minutes")

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df_melted_mean_duration, x="distance_category", y="mean_trip_duration_minutes", hue="member_casual", errorbar=None)

plt.xlabel('Distance Category')
plt.ylabel('Mean Trip Duration (Minutes)')
plt.title('Mean Trip Duration by Distance Category and Member Type')
plt.legend(title='Member Type')

# Add mean trip duration on each bar
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()

# COMMAND ----------

# Convert timestamp to date
df_citibikenyc_final_with_date = df_citibikenyc_final.withColumn("date", F.to_date("started_at"))

# Group by date and distance category, and count the number of trips
df_grouped_date = df_citibikenyc_final_with_date.groupBy("date", "distance_category").count()

# Convert Spark DataFrame to Pandas DataFrame
df_grouped_date_pd = df_grouped_date.toPandas()

# Plot the line chart using seaborn
plt.figure(figsize=(12, 8))
sns.lineplot(data=df_grouped_date_pd, x="date", y="count", hue="distance_category", marker="o")

plt.xlabel('Date')
plt.ylabel('Number of Trips')
plt.title('Number of Trips Over Time by Distance Category')
plt.legend(title='Distance Category')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC __Conclusion:__ The first visualization shows that most rides fall into the [0-2 km] and [2-4 km] categories, with the majority of riders being members. The second visualization reveals that most trips lasting more than 30 minutes are in the [4-10 km] category. The third one, while expected, confirms that average trip durations increase linearly with distance. However, it also highlights an important detail: Trips with unusually long durations are missing from the categorization due to missing end coordinates. Lastly, the final visualization indicates a correlation between distance categories on certain dates.