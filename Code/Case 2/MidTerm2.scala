/**
 * Created by shwetaanchan on 6/20/15.
 */

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}


object MidTerm2 {
  def main(args: Array[String]) {
    val sc = new SparkContext("local[2]", "xyz")

    val data = sc.textFile( "/Users/shwetaanchan/Desktop/Big Data Midterm/Income_Class.csv")

    val head = data.first()
    val data1 = data.filter(x=> x!= head)


    val labelsAndFeatures = data1.map { line =>
      val fields = line.split(',')
      if (fields(0) ==  " >50K"){
        fields(0) = "1"
      } else{
        fields(0) = "0"
      }

      LabeledPoint(fields(0).toDouble, Vectors.dense(fields.tail.map(x => x.toDouble)))
    }

    // Summary Statistics

//        val summary: MultivariateStatisticalSummary = Statistics.colStats(parsedData)
//        println(summary.mean)
//         println(summary.variance)
//        println(summary.numNonzeros)


    val scaler = new StandardScaler().fit(labelsAndFeatures.map(x => x.features))
    val normData = labelsAndFeatures.map(x => LabeledPoint(x.label, scaler.transform(x.features)))


   // PCA
    val dataSplit = normData.randomSplit(Array(0.7, 0.3), seed = 11L)
    val trainingData = dataSplit(0).cache()
    val testData = dataSplit(1)

    val pca = new PCA(50).fit(normData.map(_.features))
    val training_pca = trainingData.map(p => p.copy(features = pca.transform(p.features)))
    val test_pca = testData.map(p => p.copy(features = pca.transform(p.features)))


   // SVMWithSGD
    val numIterations = 10

    // w/o PCA
    val model_svm = SVMWithSGD.train(trainingData, numIterations)
    val prediction_nopca = testData.map{ point =>
      val prediction_svm = model_svm.predict(point.features)
      (prediction_svm, point.label)
    }
    val metrics_nopca = new BinaryClassificationMetrics(prediction_nopca)
    val auROC_nopca = metrics_nopca.areaUnderROC()



    // with PCA
    val model_svm__pca = SVMWithSGD.train(training_pca, numIterations)
    val prediction_pca = test_pca.map{ point =>
      val prediction_svm_pca = model_svm__pca.predict(point.features)
      (prediction_svm_pca, point.label)
    }
    val metrics_pca = new BinaryClassificationMetrics(prediction_pca)
    val auROC_pca = metrics_pca.areaUnderROC()


    // Decision Trees

    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32

    // w/o PCA
    val model_dectree = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)
    val  pred_nopca= testData.map { point =>
      val prediction = model_dectree.predict(point.features)
      (point.label, prediction)
    }
    val met_nopca = new BinaryClassificationMetrics(pred_nopca)
    val aROC_nopca = met_nopca.areaUnderROC()


    // with PCA
    val model_dectree_pca = DecisionTree.trainClassifier(training_pca, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)
    val pred_pca = test_pca.map { point =>
      val prediction = model_dectree_pca.predict(point.features)
      (point.label, prediction)
    }
    val met_pca = new BinaryClassificationMetrics(pred_pca)
    val aROC_pca = met_pca.areaUnderROC()




    println("Area under ROC for SVMWithSGD w/o PCA: " + auROC_nopca)
    println("Area under ROC for SVMWithSGD with PCA: " + auROC_pca)

    println("Area under ROC for Decision Trees w/o PCA: "  + aROC_nopca)
    println("Area under ROC for Decision Trees with PCA: " + aROC_pca)




  }
}
