/**
 * Created by shwetaanchan on 6/20/15.
 */

import org.apache.spark.SparkContext
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature._
import org.apache.spark.mllib.linalg.{Matrix, Vectors, SingularValueDecomposition}
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.sql.{Row, SQLContext}


object MidTerm {
  def main(args: Array[String]) {
    val sc = new SparkContext("local[2]", "xyz")

    val data = sc.textFile("/Users/shwetaanchan/Desktop/Big Data Midterm/YearPredictionMSD.csv")
    val sqlContext = new SQLContext(sc)
    val parsedData = data.map(x => Vectors.dense(x.split(',').map(_.toDouble))).cache()

    val labelsAndFeatures = data.map { line =>
      val fields = line.split(',')
      var num ="0"
      if (fields(0) >= "1965"){
        num = "1"
      }
      LabeledPoint(num.toDouble, Vectors.dense(fields.tail.map(x => x.toDouble)))
    }

    val labelsAndFeatures_reg = data.map { line =>
      val fields = line.split(',')
      LabeledPoint(fields(0).toDouble, Vectors.dense(fields.tail.map(x => x.toDouble)))
    }


    //Summary Statistics

//    val summary: MultivariateStatisticalSummary = Statistics.colStats(parsedData)
//    println(summary.mean)
//    println(summary.variance)
//    println(summary.numNonzeros)


    val scaler = new StandardScaler().fit(labelsAndFeatures.map(x => x.features))
    val normData = labelsAndFeatures.map(x => LabeledPoint(x.label, scaler.transform(x.features)))

    val scaler_reg = new StandardScaler().fit(labelsAndFeatures_reg.map(x => x.features))
    val normData_reg = labelsAndFeatures_reg.map(x => LabeledPoint(x.label, scaler_reg.transform(x.features)))


    // PCA
    val dataSplit = normData.randomSplit(Array(0.7, 0.3), seed = 11L)
    val trainingData = dataSplit(0).cache()
    val testData = dataSplit(1)

    val pca = new PCA(50).fit(normData.map(_.features))
    val training_pca = trainingData.map(p => p.copy(features = pca.transform(p.features)))
    val test_pca = testData.map(p => p.copy(features = pca.transform(p.features)))

    val dataSplit_reg = normData_reg.randomSplit(Array(0.7, 0.3), seed = 11L)
    val trainingData_reg = dataSplit_reg(0).cache()
    val testData_reg = dataSplit_reg(1)

    val pca_reg = new PCA(50).fit(normData_reg.map(_.features))
    val training_pca_reg = trainingData_reg.map(p => p.copy(features = pca_reg.transform(p.features)))
    val test_pca_reg = testData_reg.map(p => p.copy(features = pca_reg.transform(p.features)))


   // Classification: Decision Trees
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32

    val model = DecisionTree.trainClassifier(training_pca, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)
    val labelsAndPredictions = test_pca.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testError = labelsAndPredictions.filter(r => r._1 != r._2).count.toDouble / test_pca.count()

    val metric_pca = new BinaryClassificationMetrics(labelsAndPredictions)
    val aROC_pca = metric_pca.areaUnderROC()

    val model_nopca = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)
    val labelsAndPredictions_nopca = testData.map { point =>
      val prediction = model_nopca.predict(point.features)
      (point.label, prediction)
    }
    val testError_nopca = labelsAndPredictions_nopca.filter(r => r._1 != r._2).count.toDouble / testData.count()

    val metric_nopca = new BinaryClassificationMetrics(labelsAndPredictions_nopca)
    val aROC_nopca = metric_nopca.areaUnderROC()





    // Regression: Decision Trees
    val impurity1 = "variance"

    val model1 = DecisionTree.trainRegressor(training_pca_reg, categoricalFeaturesInfo, impurity1,
      maxDepth, maxBins)
    val labelsAndPredictions1 = test_pca_reg.map { point =>
      val prediction = model1.predict(point.features)
      (point.label, prediction)
    }
    val testMSE = labelsAndPredictions1.map{ i => math.pow((i._1 - i._2), 2)}.mean()

    val model1_nopca = DecisionTree.trainRegressor(trainingData_reg, categoricalFeaturesInfo, impurity1,
      maxDepth, maxBins)
    val labelsAndPredictions1_nopca = testData_reg.map { point =>
      val prediction = model1_nopca.predict(point.features)
      (point.label, prediction)
    }
    val testMSE_nopca = labelsAndPredictions1_nopca.map{ i => math.pow((i._1 - i._2), 2)}.mean()

    println("Learned classification tree model:\n" + model.toDebugString)



    // SVMWithSGD

    val numIterations = 20

    // w/o PCA
    val model_svm = SVMWithSGD.train(trainingData, numIterations)
    val prediction_nopca = testData.map{ point =>
      val prediction_svm = model_svm.predict(point.features)
      (prediction_svm, point.label)
    }
    val metrics_nopca = new BinaryClassificationMetrics(prediction_nopca)
    val auROC_nopca = metrics_nopca.areaUnderROC()

    val testError_svm = prediction_nopca.filter(r => r._1 != r._2).count.toDouble / test_pca.count()

    // with PCA
    val model_svm__pca = SVMWithSGD.train(training_pca, numIterations)
    val prediction_pca = test_pca.map{ point =>
      val prediction_svm_pca = model_svm__pca.predict(point.features)
      (prediction_svm_pca, point.label)
    }
    val metrics_pca = new BinaryClassificationMetrics(prediction_pca)
    val auROC_pca = metrics_pca.areaUnderROC()

    val testError_nopca_svm = prediction_pca.filter(r => r._1 != r._2).count.toDouble / testData.count()


    // LinearRegressionWithSGD
    val numIterationss = 20

    // w/o PCA
    val models = LinearRegressionWithSGD.train(trainingData_reg, numIterationss)
    val valuesAndPred = testData_reg.map { point =>
      val prediction = models.predict(point.features)
      (point.label, prediction)
       }
    val M = valuesAndPred.map(i => math.pow((i._1-i._2),2)).mean()


    // with PCA
    val models_pca = LinearRegressionWithSGD.train(training_pca_reg, numIterationss)
    val valuesAndPred_pca = test_pca_reg.map { point =>
      val prediction = models_pca.predict(point.features)
      (point.label, prediction)
    }
    val M_pca = valuesAndPred_pca.map(i => math.pow((i._1-i._2),2)).mean()

//    println("Area under ROC for Decision tree (Classification) w/o PCA: " + aROC_nopca)
//    println("Area under ROC for Decision tree (Classification) with PCA: " + aROC_pca)
//
//    println("Area under ROC for SVMWithSGD w/o PCA: " + auROC_nopca)
//    println("Area under ROC for SVMWithSGD with PCA: " + auROC_pca)


    println("Test Error for Decision tree (Classification) w/o PCA: " + testError_nopca)
    println("Test Error for Decision tree (Classification) with PCA: " + testError)

    println("Test Error for SVMWithSGD w/o PCA: " + testError_nopca_svm)
    println("Test Error for SVMWithSGD with PCA: " + testError_svm)


    println("Decision tree (Regression) Mean Squared Error w/o PCA: " + testMSE_nopca)
    println("Decision tree (Regression) Mean Squared Error with PCA: " + testMSE)

    println("Linear Regression Mean Squared Error w/o PCA: " + M)
    println("Linear Regression Mean Squared Error with PCA: " + M_pca)
  }
}
