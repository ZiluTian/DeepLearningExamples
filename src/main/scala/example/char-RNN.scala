package example

import scala.math
import collection.immutable.IndexedSeq
import scala.io.Source
import scala.concurrent.ExecutionContext.Implicits.global
// import scala.concurrent.Task
import scalaz.std.iterable._
import scalaz.syntax.all._
import com.thoughtworks.future._
import scala.concurrent.Await
import scala.concurrent.duration.Duration
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax
import com.thoughtworks.deeplearning.plugins.DoubleLiterals
import com.thoughtworks.deeplearning.plugins.INDArrayLiterals
import com.thoughtworks.deeplearning.plugins.CumulativeDoubleLayers
import com.thoughtworks.deeplearning.plugins.DoubleTraining
import com.thoughtworks.deeplearning.plugins.CumulativeINDArrayLayers
import com.thoughtworks.deeplearning.plugins.INDArrayWeights
import com.thoughtworks.deeplearning.plugins.Operators
import com.thoughtworks.deeplearning.plugins.Logging
import com.thoughtworks.deeplearning.plugins.Builtins
import com.thoughtworks.feature.Factory


/**
  * http://dokotta.com/demo/CharRNN.html
  * http://karpathy.github.io/2015/05/21/rnn-effectiveness/
  * https://towardsdatascience.com/recurrent-neural-networks-rnns-3f06d7653a85
  * https://gist.github.com/karpathy/d4dee566867f8291f086
  */

object charRNNExample extends App {
    // hyperparameter for this example
    trait LearningRate extends INDArrayWeights {
        val learningRate: Double
        
        trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi { this: INDArrayOptimizer =>
        override def delta: INDArray = super.delta mul learningRate
        }
        override type INDArrayOptimizer <: INDArrayOptimizerApi with Optimizer
    }

    trait Adagrad extends INDArrayWeights {
        val eps: Double
        
        trait INDArrayWeightApi extends super.INDArrayWeightApi { this: INDArrayWeight =>
        var cache: Option[INDArray] = None
        }

        override type INDArrayWeight <: INDArrayWeightApi with Weight

        trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi { this: INDArrayOptimizer =>
        private lazy val deltaLazy: INDArray = {
            import org.nd4s.Implicits._
            import weight._
            val delta0 = super.delta
            cache = Some(cache.getOrElse(Nd4j.zeros(delta0.shape: _*)) + delta0 * delta0)
            delta0 / (Transforms.sqrt(cache.get) + eps)
        }
        override def delta = deltaLazy
        }
        override type INDArrayOptimizer <: INDArrayOptimizerApi with Optimizer
    }


    val data = "DeepLearning.scala"
    val dataSize = data.size

    // vocabulary
    val vocabulary = data.toSet.toArray
    // find the index of characters in the vocabulary
    val charToIx = (for (i <- vocabulary.indices) yield (vocabulary(i), i)).toMap
    val vocabSize = vocabulary.size

    // encode each character into a vector using 1-of-k encoding
    // all zero except for a single one at the index of the char in the vocabulary
    def oneOfK(c: Char) = Nd4j.zeros(vocabSize, 1).putScalar(charToIx(c), 1)

    // Specify hyperparameters
    val hyperparameters = Factory[Adagrad with LearningRate with Builtins].newInstance(learningRate = 0.05, eps=1e-8)

    import hyperparameters.INDArrayWeight
    import hyperparameters.DoubleLayer
    import hyperparameters.INDArrayLayer
    import hyperparameters.implicits._

    /** hidden state acts as memory of the network; h(t) = f(U x(t) + W h(t-1)), 
     * f is non-linear transformation
     * U is the weight matrix that parameterizes input-to-hidden connections
     * W is the weight matrix that parameterizes hidden-to-hidden connections
     * V is the weight matrix that parameterizes hidden-to-output connections 
     * (U, V, W) are shared across time
     **/
    val hiddenSize = 100    // size of hidden layer of neurons
    val seqLength = 25      // number of steps to unroll the RNN for

    // model parameters
    // input to hidden (xh)
    val wxh = {
        import org.nd4s.Implicits._
        INDArrayWeight(Nd4j.randn(hiddenSize, vocabSize) * 0.01)
    }

    // hidden to hidden (hh)
    val whh = {
        import org.nd4s.Implicits._
        INDArrayWeight(Nd4j.randn(hiddenSize, hiddenSize) * 0.01)
    }

    // hidden to output (hy)
    val why = {
        import org.nd4s.Implicits._
        INDArrayWeight(Nd4j.randn(vocabSize, hiddenSize) * 0.01)
    }

    // hidden bias
    val bh = INDArrayWeight(Nd4j.zeros(hiddenSize, 1))
    // output bias
    val by = INDArrayWeight(Nd4j.zeros(vocabSize, 1))

    // Implement the neural network
    def tanh(x: INDArrayLayer): INDArrayLayer = {
        val exp_x = hyperparameters.exp(x)
        val exp_nx = hyperparameters.exp(-x)
        (exp_x - exp_nx) / (exp_x + exp_nx)
    }

    def charRNN(x: INDArray, y: INDArray, hprev: INDArrayLayer): (DoubleLayer, INDArrayLayer, INDArrayLayer) = {
        val hnext = tanh(wxh.dot(x) + whh.dot(hprev) + bh)
        val yraw = why.dot(hnext) + by
        val yraw_exp = hyperparameters.exp(yraw)
        val prob = yraw_exp / yraw_exp.sum
        val loss = -hyperparameters.log((prob * y).sum)
        (loss, prob, hnext)
    }

    val batches = data.zip(data.tail).grouped(seqLength).toVector

    type WithHiddenLayer[A] = (A, INDArrayLayer)
    type Batch = IndexedSeq[(Char, Char)]
    type Losses = Vector[Double]

    def singleBatch(batch: WithHiddenLayer[Batch]): WithHiddenLayer[DoubleLayer] = {
    batch match {
        case (batchseq, hprev) => batchseq.foldLeft((DoubleLayer(0.0.forward), hprev)) {
        (bstate: WithHiddenLayer[DoubleLayer], xy: (Char, Char)) =>
            (bstate, xy) match {
            case ((tot, localhprev), (x, y)) => {
                charRNN(oneOfK(x), oneOfK(y), localhprev) match {
                case (localloss, _, localhnext) => {
                    (tot + localloss, localhnext)
                }
                }
            }
            }
        }
    }
    }

    def initH = INDArrayLayer(Nd4j.zeros(hiddenSize, 1).forward)

    def singleRound(initprevloss: Losses): Future[Losses] =
    (batches.foldLeftM((initprevloss, initH)) {
        (bstate: WithHiddenLayer[Losses], batch: Batch) =>
        bstate match {
            case (prevloss, hprev) => singleBatch(batch, hprev) match {
            case (bloss, hnext) => bloss.train.map {
                (blossval: Double) => {
                    val nloss = prevloss.last * 0.999 + blossval * 0.001
                    val loss_seq = prevloss :+ prevloss.last * 0.999 + blossval * 0.001
                    (loss_seq, hnext)
                }
            }
            }
        }
    }).map {
        (fstate: WithHiddenLayer[Losses]) =>
        fstate match {
            case (floss, _) => floss
        }
    }

    def allRounds: Future[Losses] = (0 until 2048).foldLeftM(Vector(-math.log(1.0 / vocabSize) * seqLength)) {
        (ploss: Losses, round: Int) => {
            singleRound(ploss)
        }
    }

    // Train the model and generate text
    def unsafePerformFuture[A](f: Future[A]): A = Await.result(f.toScalaFuture, Duration.Inf)

    val losses = unsafePerformFuture(allRounds)

    def genIdx(v: INDArray): Int = Nd4j.getExecutioner().execAndReturn(new IMax(v)).getFinalResult()

    // a generative model
    def generate(seed: Char, n: Int): Future[String] = ((0 until n).foldLeftM((seed.toString, initH)) {
        (st: (String, INDArrayLayer), i: Int) =>
            st match {
                case (tot, hprev) => {
                    val x = oneOfK(tot.last)
                    charRNN(x, x, hprev) match {
                        case (_, prob, hnext) =>
                            prob.predict.flatMap { (probv: INDArray) =>
                                val nidx = genIdx(probv)
                                val nc = vocabulary(nidx)
                                Future.now(tot + nc.toString, hnext)
                            }
                    }
                }
            }
    }).map { (st: (String, INDArrayLayer)) =>
    st match {
        case (r, _) => r
    }
    }

    println(unsafePerformFuture(generate('D', 128)))
}