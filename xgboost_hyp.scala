//https://spark.apache.org/docs/latest/ml-tuning.html

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.tuning._
import org.apache.spark.sql._
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

// Assemble features
val inputCols = df.columns.filter{_!=targetCol}
val assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol(featuresCol)
val xgbInput = assembler.transform(df).select(featuresCol, targetCol)

// Split the data into training and test set
val Array(training, test) = xgbInput.randomSplit(Array(0.8, 0.2), 123)

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

// Model evaluation
val evaluator = new RegressionEvaluator().setLabelCol(targetCol).setPredictionCol("prediction")

def hyperparametertuningCV(booster: XGBoostRegressor, evaluator: RegressionEvaluator, training : Dataset[Row], test: Dataset[Row]): CrossValidatorModel ={
    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    val paramGrid = new ParamGridBuilder().addGrid(booster.maxDepth, Array(4, 7)).addGrid(booster.eta, Array(0.1, 0.6)).build()
    val cv = new CrossValidator().setEstimator(booster).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(5)
    val cvModel = cv.fit(training)
    return cvModel
}

val cvModel = hyperparametertuningCV(booster, evaluator, training, test)
val paramMap = cvModel.bestModel.extractParamMap
val predictions = cvModel.transform(test)
val rmse = evaluator.evaluate(predictions)
println("The model accuracy(rmse) is : " + rmse)