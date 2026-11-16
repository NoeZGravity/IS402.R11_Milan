from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("5G-BigData-Test")
    .master("local[*]")
    .getOrCreate()
)

df = spark.range(1_000_000)
print("Tong so dong:", df.count())
df.show(5)

spark.stop()