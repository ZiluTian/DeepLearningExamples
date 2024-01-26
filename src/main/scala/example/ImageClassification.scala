package example

import scala.concurrent.ExecutionContext.Implicits.global
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import com.thoughtworks.deeplearning.plugins.Builtins
import com.thoughtworks.feature.Factory
import com.thoughtworks.future._
import scala.concurrent.Await
import scala.concurrent.duration.Duration
import com.thoughtworks.each.Monadic._
import scalaz.std.stream._

// Use softmax classifier for image classification
// http://dokotta.com/demo/CharRNN.html
// BUG: runtime error in the original source code (index error, unfixed)
object ImageClassification extends App {

    // setup hyperparameters
    import scala.concurrent.ExecutionContext.Implicits.global
    import com.thoughtworks.feature.Factory
    val hyperparameters = Factory[Builtins with FixedLearningRate].newInstance(learningRate = 0.01)

    import hyperparameters.implicits._

    // Define softmax classifier
    import hyperparameters.INDArrayLayer
    def softmax(scores: INDArrayLayer): INDArrayLayer = {
        val expScores = hyperparameters.exp(scores)
        expScores / expScores.sum(1)
    }

    // Compose neural network
    //10 label of CIFAR10 images(airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck)
    val NumberOfClasses: Int = 10
    val NumberOfPixels: Int = 3072

    import hyperparameters.INDArrayWeight

    val weight = {
        import org.nd4s.Implicits._
        INDArrayWeight(Nd4j.randn(NumberOfPixels, NumberOfClasses) * 0.001)
    }

    def myNeuralNetwork(input: INDArray): INDArrayLayer = {
        softmax(input dot weight)
    }

    // Loss function
    import hyperparameters.DoubleLayer

    def lossFunction(input: INDArray, expectOutput: INDArray): DoubleLayer = {
        val probabilities = myNeuralNetwork(input)
        -(hyperparameters.log(probabilities) * expectOutput).mean
    }

    // Prepare data
    val trainNDArray = ReadCIFAR10ToNDArray.readFromResource("/cifar-10-batches-bin/data_batch_1.bin", 1000)
    val testNDArray = ReadCIFAR10ToNDArray.readFromResource("/cifar-10-batches-bin/test_batch.bin", 100)

    // Process data
    val trainData = trainNDArray.head
    val testData = testNDArray.head

    val trainExpectResult = trainNDArray.tail.head
    val testExpectResult = testNDArray.tail.head
    //zt BUG: Invalid indices: cannot get [10,10] from a [1000, 10] NDArray
    val vectorizedTrainExpectResult = Utils.makeVectorized(trainExpectResult, NumberOfClasses)
    val vectorizedTestExpectResult = Utils.makeVectorized(testExpectResult, NumberOfClasses)

    // Train neural network
    var lossSeq: IndexedSeq[Double] = IndexedSeq.empty

    @monadic[Future]
    val trainTask: Future[Unit] = {
        val lossStream = for (_ <- (1 to 2000).toStream) yield {
            lossFunction(trainData, vectorizedTrainExpectResult).train.each
        }
        lossSeq = IndexedSeq.concat(lossStream)
    }

    // Predict
    Await.result(trainTask.toScalaFuture, Duration.Inf)
    val predictResult = Await.result(myNeuralNetwork(testData).predict.toScalaFuture, Duration.Inf)

    println("The accuracy is " + Utils.getAccuracy(predictResult,testExpectResult) + "%")
}