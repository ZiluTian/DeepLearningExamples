package example

import org.nd4j.linalg.api.ndarray.INDArray

import com.thoughtworks.each.Monadic._
import com.thoughtworks.deeplearning.plugins.Builtins

import com.thoughtworks.future._
import scala.concurrent.Await
import scala.concurrent.duration.Duration
import scalaz.std.stream._

// http://dokotta.com/demo/GettingStarted.html

object HelloWorld extends App {
  val TrainingQuestions: INDArray = {
    import org.nd4s.Implicits._
    Array(
      Array(0, 1, 2),
      Array(4, 7, 10),
      Array(13, 15, 17)
    ).toNDArray
  }

  val ExpectedAnswers: INDArray = {
    import org.nd4s.Implicits._
    Array(
      Array(3),
      Array(13),
      Array(19)
    ).toNDArray
  }


  // setup hyperparameters
  import scala.concurrent.ExecutionContext.Implicits.global
  import com.thoughtworks.feature.Factory
  val hyperparameters = Factory[Builtins with FixedLearningRate].newInstance(learningRate = 0.003)

  // Build an untrained neural network of the robot
  import hyperparameters.implicits._

  def initialValueOfRobotWeight: INDArray = {
    import org.nd4j.linalg.factory.Nd4j
    import org.nd4s.Implicits._
    Nd4j.randn(3, 1)
  }

  import hyperparameters.INDArrayWeight
  val robotWeight: INDArrayWeight = INDArrayWeight(initialValueOfRobotWeight)

  import hyperparameters.INDArrayLayer
  def iqTestRobot(questions: INDArray): INDArrayLayer = {
    questions dot robotWeight
  }

  // Train the network
  // loss function
  import hyperparameters.DoubleLayer
  def squareLoss(questions: INDArray, expectAnswer: INDArray): DoubleLayer = {
    val difference = iqTestRobot(questions) - expectAnswer
    (difference * difference).mean
  }


  // Run the training task
  val TotalIterations = 500

  @monadic[Future]
  def train: Future[Stream[Double]] = {
    for (iteration <- (0 until TotalIterations).toStream) yield {
      squareLoss(TrainingQuestions, ExpectedAnswers).train.each
    }
  }

  val lossByTime: Stream[Double] = Await.result(train.toScalaFuture, Duration.Inf)

  // Test the trained robot
  val TestQuestions: INDArray = {
    import org.nd4s.Implicits._
    Array(Array(6, 9, 12)).toNDArray
  }

  val predict = Await.result(iqTestRobot(TestQuestions).predict.toScalaFuture, Duration.Inf)
  println(f"Test: What is the next number in the series ${TestQuestions}\nRobot answers: ${predict}")

  // val weightData: INDArray = robotWeight.data
  // println(weightData)
}