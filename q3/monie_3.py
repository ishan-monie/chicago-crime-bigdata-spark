from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, year, month, dayofweek, weekofyear, lag, avg as spark_avg,
    count, to_date, date_add, max as spark_max, lit
)
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import math

# ── Init ──────────────────────────────────────────────────────────────────────
spark = SparkSession.builder.appName("CrimePrediction").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

DATA_PATH = "hdfs:///user/eil8095/crime/Crimes_-_2001_to_present.csv"

# IUCR violent crime codes (FBI Part I violent crimes)
# Homicide=01xx, CSA=02xx, Robbery=03xx, Agg Assault=04xx
VIOLENT_IUCR_PREFIXES = ("01", "02", "03", "04")

# ── Load & prep ───────────────────────────────────────────────────────────────
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

df = df.select(
    col("Date"),
    col("Beat").cast("string"),
    col("IUCR").cast("string"),
    col("Arrest").cast("boolean"),
    col("Year").cast("int")
).filter(col("Beat").isNotNull() & col("Date").isNotNull())

# Parse date
df = df.withColumn("date_parsed", to_date(col("Date"), "MM/dd/yyyy hh:mm:ss a"))
df = df.filter(col("date_parsed").isNotNull())

# Flag violent crimes
df = df.withColumn(
    "is_violent",
    col("IUCR").substr(1, 2).isin(list(VIOLENT_IUCR_PREFIXES)).cast("int")
)

# Aggregate to weekly beat-level counts (model at week granularity)
df = df.withColumn("week", weekofyear(col("date_parsed"))) \
       .withColumn("yr",   year(col("date_parsed")))

weekly = df.groupBy("Beat", "yr", "week").agg(
    count("*").alias("total_crimes"),
    spark_avg("is_violent").alias("violent_ratio")
).withColumn("weighted_crimes",
    col("total_crimes") * (1 + col("violent_ratio"))  # upweight violent
)

# ── Feature engineering ───────────────────────────────────────────────────────
w = Window.partitionBy("Beat").orderBy("yr", "week")

weekly = weekly \
    .withColumn("lag1",  lag("total_crimes", 1).over(w)) \
    .withColumn("lag2",  lag("total_crimes", 2).over(w)) \
    .withColumn("lag4",  lag("total_crimes", 4).over(w)) \
    .withColumn("roll4", (lag("total_crimes",1).over(w) +
                          lag("total_crimes",2).over(w) +
                          lag("total_crimes",3).over(w) +
                          lag("total_crimes",4).over(w)) / 4) \
    .withColumn("month_num", (col("week") / 4.33).cast("int") + 1)

weekly = weekly.dropna(subset=["lag1", "lag2", "lag4", "roll4"])

# ── Train / test split (last 4 weeks = test) ──────────────────────────────────
max_yr   = weekly.agg(spark_max("yr")).collect()[0][0]
max_week = weekly.filter(col("yr") == max_yr).agg(spark_max("week")).collect()[0][0]

test_mask = (col("yr") == max_yr) & (col("week") >= max_week - 3)

train = weekly.filter(~test_mask)
test  = weekly.filter(test_mask)

print(f"Train size: {train.count():,}  |  Test size: {test.count():,}")

# ── ML Pipeline ───────────────────────────────────────────────────────────────
feature_cols = ["lag1", "lag2", "lag4", "roll4", "week", "month_num", "violent_ratio"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Model 1: GBT (better for non-linear patterns)
gbt = GBTRegressor(
    featuresCol="features",
    labelCol="weighted_crimes",
    maxIter=50,
    maxDepth=5,
    stepSize=0.1
)

# Model 2: Random Forest (more stable, good baseline)
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="weighted_crimes",
    numTrees=100,
    maxDepth=6
)

pipeline_gbt = Pipeline(stages=[assembler, gbt])
pipeline_rf  = Pipeline(stages=[assembler, rf])

evaluator_rmse = RegressionEvaluator(
    labelCol="weighted_crimes",
    predictionCol="prediction",
    metricName="rmse"
)
evaluator_r2 = RegressionEvaluator(
    labelCol="weighted_crimes",
    predictionCol="prediction",
    metricName="r2"
)

# ── Fit & evaluate ────────────────────────────────────────────────────────────
model_gbt = pipeline_gbt.fit(train)
model_rf  = pipeline_rf.fit(train)

preds_gbt = model_gbt.transform(test)
preds_rf  = model_rf.transform(test)

rmse_gbt  = evaluator_rmse.evaluate(preds_gbt)
r2_gbt  = evaluator_r2.evaluate(preds_gbt)
rmse_rf   = evaluator_rmse.evaluate(preds_rf)
r2_rf   = evaluator_r2.evaluate(preds_rf)

output_lines = [
    "Task 3 – Crime Prediction at Beat Level (Next Week)\n",
    f"Train rows : {train.count():,}",
    f"Test rows  : {test.count():,}",
    "",
    "Model Performance:",
    f"  GBT  — RMSE: {rmse_gbt:.3f}  |  MAPE: {r2_gbt:.3f}",
    f"  RF   — RMSE: {rmse_rf:.3f}  |  MAPE: {r2_rf:.3f}",
    "",
    "Sample predictions (GBT) — first 20 beats:"
]

print("\n".join(output_lines))

sample = preds_gbt.select("Beat", "yr", "week", "weighted_crimes", "prediction") \
                  .orderBy("Beat", "yr", "week").limit(20)
sample.show(truncate=False)

sample_str = "\n".join([str(r) for r in sample.collect()])
output_lines.append(sample_str)

# ── Save ──────────────────────────────────────────────────────────────────────
with open("monie_3.txt", "w") as f:
    f.write("\n".join(output_lines))

print("\nSaved to monie_3.txt")
spark.stop()
