from pyspark.sql import SparkSession
from pyspark.sql.functions import month, year, count, avg

spark = SparkSession.builder.appName("CrimeHistogram").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")
DATA_PATH = "hdfs:///user/eil8095/crime/Crimes_-_2001_to_present.csv"

df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

df.createOrReplaceTempView("crimes")

# SparkSQL: count events per (year, month)
monthly_counts = spark.sql("""
    SELECT
        YEAR(TO_DATE(Date, 'MM/dd/yyyy hh:mm:ss a')) AS yr,
        MONTH(TO_DATE(Date, 'MM/dd/yyyy hh:mm:ss a')) AS mo,
        COUNT(*) AS event_count
    FROM crimes
    WHERE Date IS NOT NULL
    GROUP BY yr, mo
""")

monthly_counts.createOrReplaceTempView("monthly_counts")

# SparkSQL: average across years for each month
avg_by_month = spark.sql("""
    SELECT
        mo AS month,
        ROUND(AVG(event_count), 2) AS avg_events
    FROM monthly_counts
    GROUP BY mo
    ORDER BY mo
""")

avg_by_month.show(12)

# ASCII histogram
rows = avg_by_month.collect()
max_val = max(r["avg_events"] for r in rows)
bar_width = 50

MONTH_NAMES = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

lines = []
lines.append("\nAverage Crime Events by Month (2001–present)\n")
lines.append(f"{'Month':<6} {'Avg Events':>10}   Histogram")
lines.append("-" * 70)
for r in rows:
    mo = int(r["month"])
    val = float(r["avg_events"])
    bar = "█" * int((val / max_val) * bar_width)
    lines.append(f"{MONTH_NAMES[mo]:<6} {val:>10.0f}   {bar}")

histogram_str = "\n".join(lines)
print(histogram_str)

# Save output
with open("monie_1.txt", "w") as f:
    f.write(histogram_str + "\n")

avg_by_month.coalesce(1).write.mode("overwrite").option(
    "header", True
).csv("hdfs:///user/eil8095/output/task1")

spark.stop()
