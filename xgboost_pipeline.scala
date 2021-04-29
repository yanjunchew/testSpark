
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
//spark2-shell --packages ml.dmlc:xgboost4j-spark:0.90

val spark = SparkSession.builder().getOrCreate()
val dataset_path = "/user/yanjunchew/diabetes_dataset.csv"
val dfraw = spark.read.option("header","true").option("inferSchema","true").csv(dataset_path)
val df0 = dfraw.withColumnRenamed("target","label")
val featuresCol = "features"
val targetCol = "label"
val columnsToDrop = Seq()

//preprocessing
val inputColsFilter = df0.columns.filter{x=> !columnsToDrop.contains(x)}
val df = {df0.select(inputColsFilter.head, inputColsFilter.tail:_*).columns.foldLeft(df0.select(inputColsFilter.head, inputColsFilter.tail:_*)){
    (df, columnName) => df.withColumn(columnName, col(columnName).cast("Double"))
  }
}

// Split the data into training and test set
val Array(training, test) = df.randomSplit(Array(0.8, 0.2), 123)

// Assemble features
val inputCols = df.columns.filter{_!=targetCol}
val assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol(featuresCol)

// Create XGBoostRegressor model
val booster = new XGBoostRegressor(
  Map(
    "learning_rate" -> 0.05,
    "gamma" -> 1,
    "objective" ->"reg:gamma",
    "subsample" -> 0.8,
    "num_round" -> 100,
    "allow_non_zero_for_missing" -> "true",
    "missing" -> 0
  )
)

// Create the training pipeline
val pipeline = new Pipeline().setStages(Array(assembler, booster))

// Train model
val model = pipeline.fit(training)

// Batch prediction
val prediction = model.transform(test)
prediction.show(true)

// Model evaluation
val evaluator = new RegressionEvaluator().setLabelCol(targetCol).setPredictionCol("prediction")

// Calculate the accuracy of the model
val accuracy = evaluator.evaluate(prediction)
println("The model accuracy is : " + accuracy)

