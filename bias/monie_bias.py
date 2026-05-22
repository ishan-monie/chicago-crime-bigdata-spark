from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, sum as spark_sum, round as spark_round,
    when, upper, trim, regexp_replace, to_date, weekofyear, year, lag
)
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("BiasFairness").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

DATA_PATH  = "hdfs:///user/eil8095/crime/Crimes_-_2001_to_present.csv"
SOCIO_PATH = "hdfs:///user/eil8095/crime/Census_Data.csv"
ACS_PATH   = "hdfs:///user/eil8095/crime/ACS_5_Year_Data.csv"

# ── load socioeconomic data ───────────────────────────────────────────────────
socio = spark.read.csv(SOCIO_PATH, header=True, inferSchema=True).select(
    col("Community Area Number").cast("int").alias("community_area_num"),
    upper(trim(col("COMMUNITY AREA NAME"))).alias("community_area_name"),
    regexp_replace(col("PERCENT HOUSEHOLDS BELOW POVERTY"), ",", "").cast("double").alias("pct_poverty"),
    regexp_replace(col("`PER CAPITA INCOME `"), ",", "").cast("double").alias("per_capita_income"),
    col("HARDSHIP INDEX").cast("double").alias("hardship_index")
).filter(col("community_area_num").isNotNull())

# ── load ACS demographic data (most recent year only) ─────────────────────────
acs_raw = spark.read.csv(ACS_PATH, header=True, inferSchema=True)
max_year = acs_raw.agg({"ACS Year": "max"}).collect()[0][0]
acs_raw = acs_raw.filter(col("ACS Year") == max_year)

acs = acs_raw.select(
    upper(trim(col("Community Area"))).alias("community_area_name"),
    regexp_replace(col("Total Population"), ",", "").cast("double").alias("total_pop"),
    regexp_replace(col("Black or African American"), ",", "").cast("double").alias("black_pop"),
    regexp_replace(col("Hispanic or Latino"), ",", "").cast("double").alias("hispanic_pop"),
    regexp_replace(col("White Not Hispanic or Latino"), ",", "").cast("double").alias("white_pop")
)

# fix name mismatches between ACS and socio
acs = acs.withColumn("community_area_name",
    when(col("community_area_name") == "MONTCLARE",          "MONTCLAIRE")
   .when(col("community_area_name") == "OHARE",              "O'HARE")
   .when(col("community_area_name") == "WASHINGTON HEIGHTS", "WASHINGTON HEIGHT")
   .otherwise(col("community_area_name"))
)

acs = acs.withColumn("pct_black",    spark_round(100.0 * col("black_pop")    / col("total_pop"), 1)) \
         .withColumn("pct_hispanic", spark_round(100.0 * col("hispanic_pop") / col("total_pop"), 1)) \
         .withColumn("pct_white",    spark_round(100.0 * col("white_pop")    / col("total_pop"), 1)) \
         .drop("total_pop","black_pop","hispanic_pop","white_pop")

# join ACS onto socio via name to get community area number
socio = socio.join(acs, on="community_area_name", how="left")

# ── load crime data ───────────────────────────────────────────────────────────
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True).select(
    col("Beat").cast("string"),
    col("Community Area").cast("int").alias("community_area_num"),
    col("Primary Type").alias("crime_type"),
    col("Arrest").cast("boolean").alias("arrest"),
    col("Date"),
    col("Year").cast("int")
).filter(col("Beat").isNotNull() & col("community_area_num").isNotNull())

# join socio+demo onto crime data via community area number
df = df.join(socio.drop("community_area_name"), on="community_area_num", how="left")

# ── group labels ──────────────────────────────────────────────────────────────
df = df.withColumn("hardship_group",
    when(col("hardship_index") <= 33, "low")
   .when(col("hardship_index") <= 66, "medium")
   .otherwise("high"))

df = df.withColumn("majority_group",
    when(col("pct_black") >= 50,    "majority_black")
   .when(col("pct_hispanic") >= 50, "majority_hispanic")
   .when(col("pct_white") >= 50,    "majority_white")
   .otherwise("mixed"))

df = df.withColumn("income_group",
    when(col("per_capita_income") <= 15000, "very_low")
   .when(col("per_capita_income") <= 25000, "low")
   .when(col("per_capita_income") <= 35000, "medium")
   .otherwise("high"))

# ── EDA ───────────────────────────────────────────────────────────────────────
print("crime volume by hardship group")
df.groupBy("hardship_group").agg(
    count("*").alias("total_crimes"),
    spark_round(avg("pct_poverty"), 1).alias("avg_pct_poverty")
).orderBy("total_crimes", ascending=False).show()

print("arrest rate by majority group")
df.groupBy("majority_group").agg(
    count("*").alias("total_crimes"),
    spark_round(100.0 * spark_sum(col("arrest").cast("int")) / count("*"), 2).alias("arrest_pct")
).orderBy("majority_group").show()

print("narcotics and violent crime share by majority group")
df.groupBy("majority_group").agg(
    count("*").alias("total"),
    spark_round(100.0 * spark_sum(when(col("crime_type") == "NARCOTICS", 1).otherwise(0)) / count("*"), 2).alias("narcotics_pct"),
    spark_round(100.0 * spark_sum(when(col("crime_type").isin("HOMICIDE","ASSAULT","BATTERY","ROBBERY"), 1).otherwise(0)) / count("*"), 2).alias("violent_pct")
).orderBy("majority_group").show()

print("arrest rate by income group")
df.groupBy("income_group").agg(
    count("*").alias("total_crimes"),
    spark_round(100.0 * spark_sum(col("arrest").cast("int")) / count("*"), 2).alias("arrest_pct")
).orderBy("income_group").show()

# ── fairness metrics ──────────────────────────────────────────────────────────
df = df.withColumn("date_parsed", to_date(col("Date"), "MM/dd/yyyy hh:mm:ss a")) \
       .withColumn("week", weekofyear(col("date_parsed"))) \
       .withColumn("yr",   year(col("date_parsed")))

weekly = df.groupBy("Beat","yr","week","majority_group","hardship_group","income_group").agg(
    count("*").alias("total_crimes"),
    avg("hardship_index").alias("hardship_index"),
    avg("per_capita_income").alias("per_capita_income")
)

w = Window.partitionBy("Beat").orderBy("yr","week")
weekly = weekly \
    .withColumn("lag1",  lag("total_crimes", 1).over(w)) \
    .withColumn("lag2",  lag("total_crimes", 2).over(w)) \
    .withColumn("lag4",  lag("total_crimes", 4).over(w)) \
    .withColumn("roll4", (lag("total_crimes",1).over(w) +
                          lag("total_crimes",2).over(w) +
                          lag("total_crimes",3).over(w) +
                          lag("total_crimes",4).over(w)) / 4)

weekly = weekly.dropna(subset=["lag1","lag2","lag4","roll4","hardship_index","per_capita_income"])

max_yr   = weekly.agg({"yr": "max"}).collect()[0][0]
max_week = weekly.filter(col("yr") == max_yr).agg({"week": "max"}).collect()[0][0]

train = weekly.filter(~((col("yr") == max_yr) & (col("week") >= max_week - 3)))
test  = weekly.filter( (col("yr") == max_yr) & (col("week") >= max_week - 3))

assembler = VectorAssembler(
    inputCols=["lag1","lag2","lag4","roll4","week","hardship_index","per_capita_income"],
    outputCol="features", handleInvalid="skip")

rf = RandomForestRegressor(featuresCol="features", labelCol="total_crimes", numTrees=50)
model = Pipeline(stages=[assembler, rf]).fit(train)
preds = model.transform(test).withColumn("residual", col("prediction") - col("total_crimes"))

evaluator = RegressionEvaluator(labelCol="total_crimes", predictionCol="prediction", metricName="rmse")

print("\nRMSE by majority group:")
for grp in ["majority_black","majority_hispanic","majority_white","mixed"]:
    s = preds.filter(col("majority_group") == grp)
    if s.count() > 0:
        print(f"  {grp}: RMSE {evaluator.evaluate(s):.3f} (n={s.count()})")

print("\nRMSE by hardship group:")
for grp in ["low","medium","high"]:
    s = preds.filter(col("hardship_group") == grp)
    if s.count() > 0:
        print(f"  {grp}: RMSE {evaluator.evaluate(s):.3f} (n={s.count()})")

print("\naverage residual by majority group (positive = overprediction):")
preds.groupBy("majority_group").agg(
    spark_round(avg("residual"), 3).alias("avg_residual"),
    count("*").alias("n")
).orderBy("majority_group").show()

print("\naverage residual by hardship group:")
preds.groupBy("hardship_group").agg(
    spark_round(avg("residual"), 3).alias("avg_residual"),
    count("*").alias("n")
).orderBy("hardship_group").show()

print("done")
spark.stop()
