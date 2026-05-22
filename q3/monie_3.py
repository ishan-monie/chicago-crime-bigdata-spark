from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, year, weekofyear, lag, avg as spark_avg,
    count, to_date, max as spark_max, regexp_replace, when
)
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("CrimePrediction").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

DATA_PATH  = "hdfs:///user/eil8095/crime/Crimes_-_2001_to_present.csv"
SOCIO_PATH = "hdfs:///user/eil8095/crime/Census_Data.csv"

# violent crime IUCR codes with Index Code = 'I' from the official CPD IUCR list
# source: https://data.cityofchicago.org/widgets/c7ck-438e
# excludes non-index crimes like simple battery, involuntary manslaughter etc.
VIOLENT_IUCR_CODES = {
    "0110","0130",
    "0261","0262","0263","0264","0265","0266","0271","0272","0273","0274","0275","0281","0291",
    "0312","0313","031A","031B","0320","0325","0326","0330","0331","0334","0337","033A","033B","0340",
    "041A","041B","0420","0430","0440","0450","0451","0452","0453","0461","0462","0479",
    "0480","0481","0482","0483","0485","0487","0488","0489","0490","0491","0492","0493",
    "0495","0496","0497","0498","0499","0510",
    "051A","051B","0520","0530","0550","0551","0552","0553","0555","0556","0557","0558",
    "1050","1055"
}

# ── load exogenous socioeconomic data ─────────────────────────────────────────
socio = spark.read.csv(SOCIO_PATH, header=True, inferSchema=True).select(
    col("Community Area Number").cast("int").alias("community_area_num"),
    regexp_replace(col("PERCENT HOUSEHOLDS BELOW POVERTY"), ",", "").cast("double").alias("pct_poverty"),
    regexp_replace(col("`PER CAPITA INCOME `"), ",", "").cast("double").alias("per_capita_income"),
    col("HARDSHIP INDEX").cast("double").alias("hardship_index")
).filter(col("community_area_num").isNotNull())

# ── load crime data ───────────────────────────────────────────────────────────
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

df = df.select(
    col("Date"),
    col("Beat").cast("string"),
    col("IUCR").cast("string"),
    col("Community Area").cast("int").alias("community_area_num"),
    col("Year").cast("int")
).filter(col("Beat").isNotNull() & col("Date").isNotNull())

# join socioeconomic data
df = df.join(socio, on="community_area_num", how="left")

# parse date and flag violent crimes
df = df.withColumn("date_parsed", to_date(col("Date"), "MM/dd/yyyy hh:mm:ss a")) \
       .filter(col("date_parsed").isNotNull()) \
       .withColumn("is_violent",
           col("IUCR").isin(list(VIOLENT_IUCR_CODES)).cast("int")) \
       .withColumn("week", weekofyear(col("date_parsed"))) \
       .withColumn("yr",   year(col("date_parsed")))

# aggregate to weekly beat level, include socioeconomic features
weekly = df.groupBy("Beat", "yr", "week").agg(
    count("*").alias("total_crimes"),
    spark_avg("is_violent").alias("violent_ratio"),
    spark_avg("pct_poverty").alias("pct_poverty"),
    spark_avg("per_capita_income").alias("per_capita_income"),
    spark_avg("hardship_index").alias("hardship_index")
).withColumn("weighted_crimes",
    col("total_crimes") * (1 + col("violent_ratio"))
)

# ── feature engineering ───────────────────────────────────────────────────────
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

weekly = weekly.dropna(subset=["lag1","lag2","lag4","roll4","pct_poverty","per_capita_income","hardship_index"])

# ── train / test split ────────────────────────────────────────────────────────
max_yr   = weekly.agg(spark_max("yr")).collect()[0][0]
max_week = weekly.filter(col("yr") == max_yr).agg(spark_max("week")).collect()[0][0]

train = weekly.filter(~((col("yr") == max_yr) & (col("week") >= max_week - 3)))
test  = weekly.filter( (col("yr") == max_yr) & (col("week") >= max_week - 3))

print(f"Train size: {train.count():,}  |  Test size: {test.count():,}")

# ── ML pipeline ───────────────────────────────────────────────────────────────
# features include lag/rolling history, seasonality, violent ratio,
# and exogenous socioeconomic features from Census data
feature_cols = [
    "lag1", "lag2", "lag4", "roll4",
    "week", "month_num", "violent_ratio",
    "pct_poverty", "per_capita_income", "hardship_index"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

gbt = GBTRegressor(featuresCol="features", labelCol="weighted_crimes", maxIter=50, maxDepth=5, stepSize=0.1)
rf  = RandomForestRegressor(featuresCol="features", labelCol="weighted_crimes", numTrees=100, maxDepth=6)

pipeline_gbt = Pipeline(stages=[assembler, gbt])
pipeline_rf  = Pipeline(stages=[assembler, rf])

evaluator_rmse = RegressionEvaluator(labelCol="weighted_crimes", predictionCol="prediction", metricName="rmse")
evaluator_r2   = RegressionEvaluator(labelCol="weighted_crimes", predictionCol="prediction", metricName="r2")

# ── fit & evaluate ────────────────────────────────────────────────────────────
model_gbt = pipeline_gbt.fit(train)
model_rf  = pipeline_rf.fit(train)

preds_gbt = model_gbt.transform(test)
preds_rf  = model_rf.transform(test)

rmse_gbt = evaluator_rmse.evaluate(preds_gbt)
r2_gbt   = evaluator_r2.evaluate(preds_gbt)
rmse_rf  = evaluator_rmse.evaluate(preds_rf)
r2_rf    = evaluator_r2.evaluate(preds_rf)

output_lines = [
    "Task 3 - Crime Prediction at Beat Level (Next Week)",
    "Exogenous dataset: Census Data - Socioeconomic Indicators by Community Area",
    f"\nTrain rows: {train.count():,}",
    f"Test rows:  {test.count():,}",
    "\nModel Performance:",
    f"  GBT: RMSE {rmse_gbt:.3f}  |  R2 {r2_gbt:.3f}",
    f"  RF:  RMSE {rmse_rf:.3f}  |  R2 {r2_rf:.3f}",
    "\nNote: MAPE metric not available in this Spark version, R2 reported instead.",
    "\nSample predictions (GBT) - first 20 beats:"
]

print("\n".join(output_lines))

sample = preds_gbt.select("Beat", "yr", "week", "weighted_crimes", "prediction") \
                  .orderBy("Beat", "yr", "week").limit(20)
sample.show(truncate=False)

sample_str = "\n".join([str(r) for r in sample.collect()])
output_lines.append(sample_str)

# ── violent vs non-violent beat evaluation ───────────────────────────────────
# classify beats as high-violence if violent_ratio > median
median_vr = weekly.approxQuantile("violent_ratio", [0.5], 0.01)[0]

test_labeled = test.withColumn("beat_type",
    when(col("violent_ratio") >= median_vr, "high_violence").otherwise("low_violence"))

preds_gbt_labeled = model_gbt.transform(test_labeled)

print("\nGBT performance by beat violence level:")
output_lines.append("\nGBT performance by beat violence level:")
for grp in ["high_violence", "low_violence"]:
    subset = preds_gbt_labeled.filter(col("beat_type") == grp)
    if subset.count() > 0:
        rmse = evaluator_rmse.evaluate(subset)
        r2   = evaluator_r2.evaluate(subset)
        line = f"  {grp}: RMSE {rmse:.3f}  |  R2 {r2:.3f}  (n={subset.count()})"
        output_lines.append(line)
        print(line)

with open("monie_3.txt", "w") as f:
    f.write("\n".join(output_lines))

print("\nSaved to monie_3.txt")
spark.stop()
