import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._
import org.apache.spark.ml.evaluation.RegressionEvaluator

//import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
//spark-shell --packages ml.dmlc:xgboost4j-spark:0.90
//spark2-shell --packages ml.dmlc:xgboost4j-spark:0.90
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
//spark-shell --conf spark.jars=C:\Spark\jars\xgboost4j-spark-0.90.jar,C:\Spark\jars\xgboost4j-0.90.jar

val spark = SparkSession.builder().getOrCreate()

//val dataset_path = "data/diabetes_dataset.csv"
val dataset_path = "/user/yanjunchew/diabetes_dataset.csv"
val targetCol = "target"
val featuresCol = "features"
val columnsToDrop = Seq()

val dfraw = spark.read.option("header","true").option("inferSchema","true").csv(dataset_path)

//preprocessing
val inputColsFilter = dfraw.columns.filter{x=> !columnsToDrop.contains(x)}
val df = {dfraw.select(inputColsFilter.head, inputColsFilter.tail:_*).columns.foldLeft(dfraw.select(inputColsFilter.head, inputColsFilter.tail:_*)){
    (df, columnName) => df.withColumn(columnName, col(columnName).cast("Double"))
  }
}

// xgboostparams
val regressorParam = Map(
  "learning_rate" -> 0.05,
  "gamma" -> 1,
  "objective" ->"reg:gamma",
  "subsample" -> 0.8,
  "num_round" -> 100,
  "allow_non_zero_for_missing" -> "true",
  "missing" -> 0
)

// Assemble features
val inputCols = df.columns.filter{_!=targetCol}
val assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol(featuresCol)
  //.setHandleInvalid("keep")

val xgbInput = assembler.transform(df).select(featuresCol, targetCol)

// Create regression model
val xgbRegressor = new XGBoostRegressor(regressorParam).setFeaturesCol(featuresCol).setLabelCol(targetCol)
val model = xgbRegressor.fit(xgbInput)

// Predict and Evaluate
val prediction = model.transform(xgbInput)
val evaluator = new RegressionEvaluator().setLabelCol(targetCol).setPredictionCol("prediction")
val accuracy  = evaluator.evaluate(prediction)
println("The model accuracy is : " + accuracy)