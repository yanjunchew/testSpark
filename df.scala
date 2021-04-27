import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()

val df = spark.read.option("header","true").option("inferSchema","true").csv("data/CitiGroup.csv")

df.printSchema()

/////////////////////////
/////////////////////////

import spark.implicits._

//df.filter($"Close" > 50 && $"High" > 40).show()
//val ch_low = df.filter("Close > 50 AND High > 40").collect()

df.filter("High == 46.82").show()

df.select(corr("High","Low")).show()