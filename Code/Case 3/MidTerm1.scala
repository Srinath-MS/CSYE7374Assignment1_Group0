/**
 * Created by shwetaanchan on 6/20/15.
 */

import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.{GaussianMixture, KMeans}
import org.apache.spark.mllib.feature._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils


object MidTerm1 {
  def main(args: Array[String]) {
    val sc = new SparkContext("local[2]", "xyz")

    val data = MLUtils.loadLibSVMFile(sc,
            "/Users/shwetaanchan/Desktop/Big Data Midterm/TV_News_Channel_Commercial_Detection_Dataset/BBC.txt")

    val scaler = new StandardScaler().fit(data.map(x => x.features))
    val normData = data.map(x => LabeledPoint(x.label, scaler.transform(x.features)))

    val discretizedData = normData.map { lp =>
      LabeledPoint(lp.label, Vectors.dense(lp.features.toArray.map { x => x/16 } ) )
    }

    val selector = new ChiSqSelector(30)
    val transformer = selector.fit(data)
    val filteredData = discretizedData.map { lp =>
      LabeledPoint(lp.label, transformer.transform(lp.features))
    }

    val dataSplit = filteredData.randomSplit(Array(0.7, 0.3), seed = 11L)
    val trainingData = dataSplit(0).cache()
    val testData = dataSplit(1)


    //PCA
    val pca = new PCA(20).fit(filteredData.map(_.features))
    val training_pca = trainingData.map(p => p.copy(features = pca.transform(p.features)))
    val test_pca = testData.map(p => p.copy(features = pca.transform(p.features)))


    val pca_features_train = training_pca.map(x => x.features)
    val pca_features_test = test_pca.map(x => x.features)

    val features_train = trainingData.map(x => x.features)
    val features_test = testData.map(x => x.features)


    //K-means
    val numIterations = 10

    // with PCA
    val pca_clusters = KMeans.train(pca_features_train, 10, numIterations)
    val pca_pred = pca_clusters.predict(pca_features_test)
    val pca_cost = pca_clusters.computeCost(pca_features_test)

    // w/o PCA
    val nopca_clusters = KMeans.train(features_train, 10, numIterations)
    val nopca_pred = nopca_clusters.predict(features_test)
    val nopca_cost = nopca_clusters.computeCost(features_test)

    println("Within Set Sum of Squared Errors w/o PCA = " + nopca_cost)
    println("Within Set Sum of Squared Errors with PCA = " + pca_cost)







    //Gaussian mixture

//    val gmm = new GaussianMixture().setK(5).run(pca_features_train)
//    for (i <- 0 until gmm.k) {
//      println("weight= %f\nmu= %s\nsigma= \n%s\n" format
//        (gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma))
//    }






















  }
}
