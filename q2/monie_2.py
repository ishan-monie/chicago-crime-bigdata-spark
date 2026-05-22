from pyspark.sql import SparkSession
from datetime import datetime
from itertools import combinations
import math

spark = SparkSession.builder.appName("CrimeRDDs").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

DATA_PATH = "hdfs:///user/eil8095/crime/Crimes_-_2001_to_present.csv"

raw = sc.textFile(DATA_PATH)
header = raw.first()
rows = raw.filter(lambda line: line != header)

def parse_line(line):
    fields = []
    in_quotes = False
    current = []
    for ch in line:
        if ch == '"':
            in_quotes = not in_quotes
        elif ch == ',' and not in_quotes:
            fields.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    fields.append(''.join(current).strip())
    return fields

parsed = rows.map(parse_line).filter(lambda f: len(f) >= 17).cache()

# part 1 - top 10 blocks last 3 years
last3 = parsed.filter(lambda f: f[17].strip().isdigit() and int(f[17].strip()) >= 2018)

block_counts = (last3
    .map(lambda f: (f[3][:8], 1))
    .reduceByKey(lambda a, b: a + b)
    .sortBy(lambda x: x[1], ascending=False)
    .take(10))

output_lines = ["top 10 blocks by crime events (2018-2020)\n"]
print("top 10 blocks (2018-2020)")
for i, (block, cnt) in enumerate(block_counts, 1):
    line = f"{i}. {block} — {cnt}"
    output_lines.append(line)
    print(line)

# part 2 - beat correlation last 5 years
last5 = parsed.filter(lambda f: f[17].strip().isdigit() and int(f[17].strip()) >= 2016)

beat_year_counts = (last5
    .filter(lambda f: f[10].strip() != '')
    .map(lambda f: ((f[10].strip(), f[17].strip()), 1))
    .reduceByKey(lambda a, b: a + b))

beat_year_dict = {}
for (beat, yr), cnt in beat_year_counts.collect():
    if beat not in beat_year_dict:
        beat_year_dict[beat] = {}
    beat_year_dict[beat][yr] = cnt

years = [str(y) for y in range(2016, 2021)]

def pearson(beat_a, beat_b, data, years):
    va = [data[beat_a].get(y, 0) for y in years]
    vb = [data[beat_b].get(y, 0) for y in years]
    n = len(years)
    mean_a = sum(va) / n
    mean_b = sum(vb) / n
    num = sum((va[i]-mean_a)*(vb[i]-mean_b) for i in range(n))
    den = math.sqrt(sum((va[i]-mean_a)**2 for i in range(n)) *
                    sum((vb[i]-mean_b)**2 for i in range(n)))
    return num / den if den != 0 else 0

valid_beats = [b for b, yrs in beat_year_dict.items() if len(yrs) == 5]
beat_pairs = sc.parallelize(list(combinations(valid_beats[:200], 2)))
beat_year_bc = sc.broadcast(beat_year_dict)

correlations = (beat_pairs
    .map(lambda pair: (pair, pearson(pair[0], pair[1], beat_year_bc.value, years)))
    .sortBy(lambda x: x[1], ascending=False)
    .take(20))

output_lines.append("\nbeat pair correlations (2016-2020)\n")
print("\nbeat pair correlations (2016-2020)")
for (ba, bb), corr in correlations:
    line = f"beats {ba} and {bb}: {corr:.4f}"
    output_lines.append(line)
    print(line)

# part 3 - daley vs emanuel
daley_years   = set(str(y) for y in range(2001, 2012))
emanuel_years = set(str(y) for y in range(2012, 2020))

daley   = parsed.filter(lambda f: f[17].strip() in daley_years)
emanuel = parsed.filter(lambda f: f[17].strip() in emanuel_years)

def by_type(rdd):
    return (rdd
        .map(lambda f: (f[5].strip(), 1))
        .reduceByKey(lambda a, b: a + b)
        .sortBy(lambda x: x[1], ascending=False)
        .take(10))

daley_type   = by_type(daley)
emanuel_type = by_type(emanuel)

output_lines.append("\ndaley (2001-2011) top crime types\n")
print("\ndaley (2001-2011)")
for ctype, cnt in daley_type:
    line = f"  {ctype}: {cnt:,}"
    output_lines.append(line)
    print(line)

output_lines.append("\nemanuel (2012-2019) top crime types\n")
print("\nemanuel (2012-2019)")
for ctype, cnt in emanuel_type:
    line = f"  {ctype}: {cnt:,}"
    output_lines.append(line)
    print(line)

daley_total   = daley.count()
emanuel_total = emanuel.count()

summary = [
    f"\ndaley total (11 yrs): {daley_total:,} — avg {daley_total//11:,}/yr",
    f"emanuel total (8 yrs): {emanuel_total:,} — avg {emanuel_total//8:,}/yr",
]
for s in summary:
    output_lines.append(s)
    print(s)

with open("monie_2.txt", "w") as f:
    f.write("\n".join(output_lines))

print("\ndone, saved to monie_2.txt")
spark.stop()
