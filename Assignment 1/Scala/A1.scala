/**
 * Created by shwetaanchan on 6/20/15.
 */

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression._


object A1 {
  def main(args: Array[String]) {
    val sc = new SparkContext("local[2]", "xyz")

    // Load and parse the data file
    val data = sc.textFile("/Users/shwetaanchan/Desktop/WineQuality.csv")

    // Skipping the header
    val head = data.first()
    val data1 = data.filter(x => x != head)

    // Making "quality" column the label and rest become the features (reordering column)
    val parsedData = data1.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(11).toDouble, Vectors.dense(parts.tail.map(x => x.toDouble)))
    }

    // Split data into training set and test set
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the model
    val numIterations = 20
    val model = LinearRegressionWithSGD.train(training, numIterations)
    val model1 = RidgeRegressionWithSGD.train(training, numIterations)
    val model2 = LassoWithSGD.train(training, numIterations)

    // Evaluate model on test data
    val valuesAndPreds = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    val valuesAndPreds1 = test.map { point =>
      val prediction1 = model1.predict(point.features)
      (point.label, prediction1)
    }

    val valuesAndPreds2 = test.map { point =>
      val prediction2 = model2.predict(point.features)
      (point.label, prediction2)
    }

    // Computer Mean Squared Error for each case
    val MSE = valuesAndPreds.map{case (v, p) => math.pow((v - p), 2) }.mean()
    val MSE1 = valuesAndPreds1.map { case (v, p) => math.pow((v - p), 2) }.mean()
    val MSE2 = valuesAndPreds2.map { case (v, p) => math.pow((v - p), 2) }.mean()

    println("Mean Squared Error for Linear regression model with no regularization using SGD is " + MSE)
    println("Mean Squared Error for Linear Regression model with L2 regularization (Ridge Regression) is " + MSE1)
    println("Mean Squared Error for Linear Regression model with L1 regularization (Lasso) is " + MSE2)
  }
}
