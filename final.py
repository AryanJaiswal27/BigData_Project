from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, lag, to_timestamp, avg, when
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import sys

# -------------------------------
# Output to BOTH terminal + file
# -------------------------------
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger("full_output_log.txt")
sys.stderr = sys.stdout

# -------------------------------
# Pretty Print
# -------------------------------
def print_box(title):
    print("\n" + "="*60)
    print("[ " + title + " ]")
    print("="*60)

# -------------------------------
# Spark (optimized)
# -------------------------------
spark = SparkSession.builder \
    .appName("Energy Prediction Model - Advanced") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# -------------------------------
# STEP 1: Load
# -------------------------------
print_box("STEP 1: Loading Dataset")

df = spark.read.csv("dataset/CC_LCL-FullData.csv", header=True, inferSchema=True)
df = df.limit(300000)

print("Loaded rows:", df.count())

# -------------------------------
# STEP 2: Cleaning
# -------------------------------
print_box("STEP 2: Cleaning")

df = df.withColumn("KWH", col("KWH/hh (per half hour) ").cast("double"))
df = df.withColumn("DateTime", to_timestamp("DateTime"))
df = df.dropna()

print("After cleaning:", df.count())

# -------------------------------
# STEP 3: Reduce
# -------------------------------
print_box("STEP 3: Reduce Dataset")

df = df.limit(150000)

print("Reduced rows:", df.count())

# -------------------------------
# STEP 4: Lag Features (IMPROVED)
# -------------------------------
print_box("STEP 4: Lag Features")

windowSpec = Window.partitionBy("LCLid").orderBy("DateTime")

df = df.withColumn("lag1", lag("KWH", 1).over(windowSpec))
df = df.withColumn("lag2", lag("KWH", 2).over(windowSpec))
df = df.withColumn("lag3", lag("KWH", 3).over(windowSpec))
df = df.withColumn("lag4", lag("KWH", 4).over(windowSpec))

# -------------------------------
# STEP 5: Time Features
# -------------------------------
print_box("STEP 5: Time Features")

df = df.withColumn("hour", hour("DateTime"))
df = df.withColumn("day", dayofweek("DateTime"))

# Weekend feature (IMPORTANT)
df = df.withColumn("is_weekend", when(col("day").isin([1,7]), 1).otherwise(0))

# -------------------------------
# STEP 6: Rolling Average (TREND)
# -------------------------------
print_box("STEP 6: Rolling Feature")

rolling_window = Window.partitionBy("LCLid").orderBy("DateTime").rowsBetween(-3, 0)

df = df.withColumn("rolling_avg", avg("KWH").over(rolling_window))

# -------------------------------
# STEP 7: Drop NULLs
# -------------------------------
print_box("STEP 7: Drop NULLs")

df = df.dropna()

print("Remaining rows:", df.count())

# -------------------------------
# STEP 8: Feature Vector
# -------------------------------
print_box("STEP 8: Features")

assembler = VectorAssembler(
    inputCols=[
        "lag1", "lag2", "lag3", "lag4",
        "hour", "day", "is_weekend",
        "rolling_avg"
    ],
    outputCol="features"
)

df = assembler.transform(df)

df.select("features", "KWH").show(5, False)

# -------------------------------
# STEP 9: Train-Test Split
# -------------------------------
print_box("STEP 9: Train-Test Split")

train = df.sample(0.8, seed=42)
test = df.subtract(train)

print("Train:", train.count())
print("Test:", test.count())

# -------------------------------
# STEP 10: Train Model (TUNED)
# -------------------------------
print_box("STEP 10: Training Model")

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="KWH",
    numTrees=10,      # increased
    maxDepth=10       # deeper trees
)

model = rf.fit(train)

print("Model trained successfully")

# -------------------------------
# STEP 11: Predictions
# -------------------------------
print_box("STEP 11: Predictions")

predictions = model.transform(test)

predictions.select("DateTime", "KWH", "prediction").show(10)

# -------------------------------
# STEP 12: Evaluation
# -------------------------------
print_box("STEP 12: Evaluation")

evaluator = RegressionEvaluator(
    labelCol="KWH",
    predictionCol="prediction",
    metricName="rmse"
)

rmse = evaluator.evaluate(predictions)

print("RMSE:", rmse)

# -------------------------------
# STEP 13: Save Output
# -------------------------------
print_box("STEP 13: Saving Output")

predictions.cache()
predictions.count()

predictions.select("DateTime", "KWH", "prediction") \
    .coalesce(1) \
    .write.mode("overwrite").csv("predictions_output")

with open("model_report.txt", "w", encoding="utf-8") as f:
    f.write("Model Evaluation Report\n")
    f.write("=======================\n")
    f.write(f"RMSE: {rmse}\n")

print("Model report saved successfully!")

# -------------------------------
# DONE
# -------------------------------
print_box("PROCESS COMPLETED")

spark.stop()