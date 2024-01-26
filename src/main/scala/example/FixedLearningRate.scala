package example

trait FixedLearningRate extends com.thoughtworks.deeplearning.plugins.INDArrayWeights {

  import org.nd4s.Implicits._
  import org.nd4j.linalg.api.ndarray.INDArray

  def learningRate: Double

  trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi { this: INDArrayOptimizer =>

    private lazy val delta0: INDArray = super.delta * learningRate

    override def delta: INDArray = delta0
  }

  override type INDArrayOptimizer <: Optimizer with INDArrayOptimizerApi

}