/**
 * Created by shwetaanchan on 6/20/15.
 */

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object A3 {
  def main(args: Array[String]) {
    val sc = new SparkContext("local[2]", "xyz")

    // Load and parse the data file
    val data = sc.textFile("/Users/shwetaanchan/Desktop/WineQuality.csv")

    // Skipping the header
    val head = data.first()
    val data1 = data.filter(x=> x!= head)

    // Making classification the label (reordering column)
    val parsedData = data1.map{ line =>
       val parts = line.split(',')
       LabeledPoint(parts(11).toDouble, Vectors.dense(parts.tail.map(x => x.toDouble)))
       }

    // Split data into training set and test set
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build model
    val model = new LogisticRegressionWithLBFGS().setNumClasses(10).run(training)

    // Compute raw scores on the test set
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    // Get evaluation metrics
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision
    println("Precision for Logistic Regression With LBGFS using L2 regularization is " + precision)
}
}
