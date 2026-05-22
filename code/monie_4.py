from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, hour, dayofweek, month, count, avg,
    to_timestamp, when, round as spark_round
)

# ── Init ──────────────────────────────────────────────────────────────────────
spark = SparkSession.builder.appName("CrimeArrestPatterns").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

DATA_PATH = "hdfs:///user/eil8095/crime/Crimes_-_2001_to_present.csv"

# ── Load ──────────────────────────────────────────────────────────────────────
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

df = df.select(
    col("Date"),
    col("Arrest").cast("boolean").alias("arrest"),
    col("Primary Type").alias("crime_type")
).filter(col("Date").isNotNull() & col("arrest").isNotNull())

df = df.withColumn("ts", to_timestamp(col("Date"), "MM/dd/yyyy hh:mm:ss a")) \
       .filter(col("ts").isNotNull())

# Time features
df = df.withColumn("hr",  hour(col("ts"))) \
       .withColumn("dow", dayofweek(col("ts"))) \
       .withColumn("mo",  month(col("ts")))

# Time of day buckets
df = df.withColumn("time_of_day",
    when(col("hr").between(6,  11), "Morning")
   .when(col("hr").between(12, 17), "Afternoon")
   .when(col("hr").between(18, 21), "Evening")
   .otherwise("Night")
)

# Day of week labels (Spark: 1=Sun, 7=Sat)
DOW_LABELS = {1:"Sun",2:"Mon",3:"Tue",4:"Wed",5:"Thu",6:"Fri",7:"Sat"}
dow_expr = col("dow")  # numeric is fine for analysis

df.createOrReplaceTempView("crimes")

output_lines = []

# ── Pattern 1: Arrest rate by hour of day ────────────────────────────────────
print("\n=== Arrest Rate by Hour of Day ===")
by_hour = spark.sql("""
    SELECT hr,
           COUNT(*) AS total,
           SUM(CAST(arrest AS INT)) AS arrests,
           ROUND(100.0 * SUM(CAST(arrest AS INT)) / COUNT(*), 2) AS arrest_pct
    FROM crimes
    GROUP BY hr
    ORDER BY hr
""")
by_hour.show(24)
output_lines.append("=== Arrest Rate by Hour of Day ===")
output_lines.append("\n".join([str(r) for r in by_hour.collect()]))

# ── Pattern 2: Arrest rate by day of week ────────────────────────────────────
print("\n=== Arrest Rate by Day of Week ===")
by_dow = spark.sql("""
    SELECT dow,
           CASE dow WHEN 1 THEN 'Sun' WHEN 2 THEN 'Mon' WHEN 3 THEN 'Tue'
                    WHEN 4 THEN 'Wed' WHEN 5 THEN 'Thu' WHEN 6 THEN 'Fri'
                    ELSE 'Sat' END AS day_name,
           COUNT(*) AS total,
           SUM(CAST(arrest AS INT)) AS arrests,
           ROUND(100.0 * SUM(CAST(arrest AS INT)) / COUNT(*), 2) AS arrest_pct
    FROM crimes
    GROUP BY dow
    ORDER BY dow
""")
by_dow.show()
output_lines.append("\n=== Arrest Rate by Day of Week ===")
output_lines.append("\n".join([str(r) for r in by_dow.collect()]))

# ── Pattern 3: Arrest rate by month ──────────────────────────────────────────
print("\n=== Arrest Rate by Month ===")
by_month = spark.sql("""
    SELECT mo,
           COUNT(*) AS total,
           SUM(CAST(arrest AS INT)) AS arrests,
           ROUND(100.0 * SUM(CAST(arrest AS INT)) / COUNT(*), 2) AS arrest_pct
    FROM crimes
    GROUP BY mo
    ORDER BY mo
""")
by_month.show(12)
output_lines.append("\n=== Arrest Rate by Month ===")
output_lines.append("\n".join([str(r) for r in by_month.collect()]))

# ── Pattern 4: Arrest rate by time of day ────────────────────────────────────
print("\n=== Arrest Rate by Time of Day ===")
by_tod = spark.sql("""
    SELECT time_of_day,
           COUNT(*) AS total,
           SUM(CAST(arrest AS INT)) AS arrests,
           ROUND(100.0 * SUM(CAST(arrest AS INT)) / COUNT(*), 2) AS arrest_pct
    FROM crimes
    GROUP BY time_of_day
    ORDER BY arrest_pct DESC
""")
by_tod.show()
output_lines.append("\n=== Arrest Rate by Time of Day ===")
output_lines.append("\n".join([str(r) for r in by_tod.collect()]))

# ── Pattern 5: Top crime types by arrest rate ─────────────────────────────────
print("\n=== Top Crime Types by Arrest Rate (min 1000 events) ===")
by_type = spark.sql("""
    SELECT crime_type,
           COUNT(*) AS total,
           SUM(CAST(arrest AS INT)) AS arrests,
           ROUND(100.0 * SUM(CAST(arrest AS INT)) / COUNT(*), 2) AS arrest_pct
    FROM crimes
    GROUP BY crime_type
    HAVING COUNT(*) >= 1000
    ORDER BY arrest_pct DESC
    LIMIT 15
""")
by_type.show(truncate=False)
output_lines.append("\n=== Top Crime Types by Arrest Rate (min 1000 events) ===")
output_lines.append("\n".join([str(r) for r in by_type.collect()]))

# ── Pattern 6: Heatmap data — dow x hour arrest rate ─────────────────────────
print("\n=== Arrest Rate Heatmap: Day x Hour ===")
heatmap = spark.sql("""
    SELECT dow, hr,
           ROUND(100.0 * SUM(CAST(arrest AS INT)) / COUNT(*), 2) AS arrest_pct
    FROM crimes
    GROUP BY dow, hr
    ORDER BY dow, hr
""")
output_lines.append("\n=== Arrest Rate Heatmap: Day x Hour ===")
output_lines.append("\n".join([str(r) for r in heatmap.collect()]))

# ── Save ──────────────────────────────────────────────────────────────────────
with open("monie_4.txt", "w") as f:
    f.write("\n".join(output_lines))

print("\nSaved to monie_4.txt")
spark.stop()
