/**
 * Created by shwetaanchan on 6/20/15.
 */

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.optimization.L1Updater

object A2 {
  def main(args: Array[String]) {
    val sc = new SparkContext("local[2]", "xyz")

    // Load and parse the data file
    val data = sc.textFile("/Users/shwetaanchan/Desktop/WineQuality.csv")

    // Skipping the header
    val head = data.first()
    val data1 = data.filter(x=> x!= head)

    // Making "classification" column the label and rest the features (reordering column)
    val parsedData = data1.map{ line =>
      val parts = line.split(',')
      LabeledPoint(parts(12).toDouble, Vectors.dense(parts.tail.map(x => x.toDouble)))
    }

    // Split data into training set and test set
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the model
    val numIterations = 20
    val model_svm = SVMWithSGD.train(training, numIterations)
    val model_logReg = LogisticRegressionWithSGD.train(training, numIterations)

    // Creating models for each algorithm using L1 regularization
    val svmAlg = new SVMWithSGD()
    svmAlg.optimizer.setNumIterations(20).setRegParam(0.1).setUpdater(new L1Updater)
    val model_svmL1 = svmAlg.run(parsedData)

    val logRegAlg = new LogisticRegressionWithSGD()
    logRegAlg.optimizer.setNumIterations(20).setRegParam(0.1).setUpdater(new L1Updater)
    val model_logRegL1 = logRegAlg.run(parsedData)

    // Clearing the default threshold
    model_svm.clearThreshold()
    model_logReg.clearThreshold()


    //Evaluate model on test data for SVMWithSGD (L1 & L2)
    val labelAndPreds_svm = test.map{ point =>
       val prediction_svm = model_svm.predict(point.features)
       (prediction_svm, point.label)
    }

    val labelAndPreds_svmL1 = test.map { point =>
      val prediction_svmL1 = model_svmL1.predict(point.features)
      (prediction_svmL1, point.label)
    }

    //Evaluate model on test data for LogisticRegressionWithSGD (L1 & L2)
    val labelAndPreds_logReg = test.map{ point =>
      val prediction_logReg = model_logReg.predict(point.features)
      (prediction_logReg, point.label )
    }

     val labelAndPreds_logRegL1 = test.map{ point =>
      val prediction_logRegL1 = model_logRegL1.predict(point.features)
      (prediction_logRegL1, point.label)
    }


    // Computing Training Error for each algorithm (L1 & L2)
    val trainErr_svm = labelAndPreds_svm.filter { case (v, p) => v != p}.count.toDouble /training.count
    val trainErr_svmL1 = labelAndPreds_svmL1.filter { case (v, p) => v != p}.count.toDouble /training.count

    val trainErr_logReg = labelAndPreds_logReg.filter { case (v, p) => v != p}.count.toDouble /training.count
    val trainErr_logRegL1 = labelAndPreds_logRegL1.filter { case (v, p) => v != p}.count.toDouble /training.count


    // Computing evaluation metrics for each algorithm
    val metrics_SVM  = new BinaryClassificationMetrics(labelAndPreds_svm)
    val metrics1_LogReg  = new BinaryClassificationMetrics(labelAndPreds_logReg)

    // Calculating ROC
    val auROC_SVM = metrics_SVM.areaUnderROC()
    val auROC1_LogReg = metrics1_LogReg.areaUnderROC()

    println("Training Error for SVMWithSGD L2 regularization is " + trainErr_svm)
    println("Training Error for SVMWithSGD L1 regularization is " + trainErr_svmL1)

    println("Training Error for LogisticRegressionWithSGD L2 regularization is " + trainErr_logReg)
    println("Training Error for LogisticRegressionWithSGD L1 regularization is " + trainErr_logRegL1)

    println("Area under ROC for SVMWithSGD is " + auROC_SVM)
    println("Area under ROC for LogisiticRegressionWithSGD is " + auROC1_LogReg)
  }
}
