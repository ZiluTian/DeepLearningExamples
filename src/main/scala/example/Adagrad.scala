package example

trait Adagrad extends com.thoughtworks.deeplearning.plugins.INDArrayWeights {

  import org.nd4j.linalg.api.ndarray.INDArray
  import org.nd4j.linalg.factory.Nd4j
  import org.nd4j.linalg.ops.transforms.Transforms

  def eps: Double

  trait INDArrayWeightApi extends super.INDArrayWeightApi { this: INDArrayWeight =>
    var cache: Option[INDArray] = None
  }

  override type INDArrayWeight <: INDArrayWeightApi with Weight

  trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi { this: INDArrayOptimizer =>
    private lazy val delta0: INDArray = {
      import org.nd4s.Implicits._
      import weight._
      val superDelta = super.delta
      val newCache = weight.synchronized {
        val newCache = weight.cache.getOrElse(Nd4j.zeros(superDelta.shape: _*)) + superDelta * superDelta
        weight.cache = Some(newCache)
        newCache
      }
      superDelta / (Transforms.sqrt(newCache) + eps)
    }
    override def delta = delta0
  }
  override type INDArrayOptimizer <: INDArrayOptimizerApi with Optimizer
}