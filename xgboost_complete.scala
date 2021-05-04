
import org.apache.spark.sql._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.log4j.Logger

import ml.dmlc.xgboost4j.scala.spark._
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
//spark2-shell --packages ml.dmlc:xgboost4j-spark:0.90

@transient lazy val logger = Logger.getLogger(getClass.getName)

val spark = SparkSession.builder().getOrCreate()
val featuresCol = "features"
val targetCol = "label"

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

def scaleDF(df:DataFrame, withStd:Boolean, withMean:Boolean): DataFrame = {
  val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(withStd).setWithMean(withMean)
  val scalerModel = scaler.fit(df)
  val scaledData = scalerModel.transform(df)
  return scaledData
}

def replaceNullDf(df:DataFrame, map: Map[String, Any]): DataFrame = {
  return df.na.fill(map)
}

def preprocessing(scale:List[Boolean], replaceNull: Boolean): Array[DataFrame] = {
  val dataset_path = "/user/yanjunchew/diabetes_dataset.csv"
  val dfraw = spark.read.option("header","true").option("inferSchema","true").csv(dataset_path)
  var df = dfraw.withColumnRenamed("target","label")

  if(replaceNull == true){
    //create map to replace Null
    val meanAge = df.select(mean("Age")).collect()
    val map = Map("s2" -> 0, "age" -> meanAge(0)(0))
    df = replaceNullDf(df, map)
  }
  //FUTURE: To consider StringIndexer or Categorial for string features
  // Assemble features
  val inputCols = df.columns.filter{_!=targetCol}
  val assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol(featuresCol)
  var xgbInput = assembler.transform(df).select(featuresCol, targetCol)
  if(scale(0)==true){
    xgbInput = scaleDF(xgbInput, scale(1), scale(2))
  }
  // Split the data into training and test set
  val Array(training, test) = xgbInput.randomSplit(Array(0.8, 0.2), 123)
  return Array(training, test)
}

def timeFit(cv : CrossValidator, training: DataFrame): CrossValidatorModel = {
  // Fit the Pipeline
  val startTime = System.nanoTime()
  val model = cv.fit(training)
  val elapsedTime = (System.nanoTime() - startTime) / 1e9
  println(s"CrossValidation time: $elapsedTime seconds")
  logger.info(s"CrossValidation time: $elapsedTime seconds")
  return model
}

def hyperparametertuningCV(booster: XGBoostRegressor, evaluator: RegressionEvaluator, training : Dataset[Row], test: Dataset[Row]): CrossValidatorModel ={
  //TODO: Update GridSearch Params here
  // We use a ParamGridBuilder to construct a grid of parameters to search over.
  val paramGrid = new ParamGridBuilder().addGrid(booster.subsample, Array(0.3,0.6)).addGrid(booster.maxDepth, Array(4, 7)).addGrid(booster.eta, Array(0.1, 0.6)).build()
  val cv = new CrossValidator().setEstimator(booster).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(5)
  val cvModel = timeFit(cv, training)
  return cvModel
}

def logParams(bestParams: Map[String, Any]): Unit = {
  for((k,v)<-bestParams){
    logger.info(s"$k : $v")
  }
}

// Model evaluation
def buildXgboostModelWithCrossValidation(training: DataFrame, test: DataFrame): CrossValidatorModel = {
  val evaluator = new RegressionEvaluator().setLabelCol(targetCol).setPredictionCol("prediction")
  val cvModel = hyperparametertuningCV(booster, evaluator, training, test)
  val bestParams = cvModel.bestModel.asInstanceOf[XGBoostRegressionModel].MLlib2XGBoostParams
  logParams(bestParams)
  return cvModel
}

def logmetrics(predictions:DataFrame): Unit = {
  // Instantiate metrics object
  val metrics = new RegressionMetrics(predictions.select("prediction","label").rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))
  val rmse = metrics.rootMeanSquaredError
  println(s"RMSE: $rmse")

  logger.info("Test Metrics")
  logger.info("Test MSE:")
  logger.info(metrics.meanSquaredError)
  logger.info("Test RMSE:")
  logger.info(metrics.rootMeanSquaredError)
  logger.info("Test MAE:")
  logger.info(metrics.meanAbsoluteError)
}

def predictAndLog(bestModel: Model[_], test: DataFrame): DataFrame = {
  val predictions = bestModel.transform(test)
  logmetrics(predictions)
  return predictions
}

// preprocess data into train and test sets
val withStd = true
val withMean = false
val Array(training, test) = preprocessing(List(false,withStd,withMean), false)

// build model with cross validation and hyperparameter tuning and save model
val cvModel = buildXgboostModelWithCrossValidation(training, test)

//generate predictions and log metrics
val prediction = predictAndLog(cvModel.bestModel, test)
prediction.show(2,false)
