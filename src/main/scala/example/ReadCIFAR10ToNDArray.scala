package example

import java.io.{FileInputStream, InputStream}
import java.io._
// import sys.process._

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import shapeless._

import scala.collection.immutable.IndexedSeq

// https://raw.githubusercontent.com/ThoughtWorksInc/DeepLearning.scala-website/v1.0.0-doc/ipynbs/ReadCIFAR10ToNDArray.sc
object ReadCIFAR10ToNDArray {

  /**
    * 原始文件字节
    */
  lazy val originalFileBytesArray: Array[Array[Byte]] = {
    for (fileIndex <- 1 to 5) yield {
      val fileName = "/cifar-10-batches-bin" + "/data_batch_" + fileIndex + ".bin"
      val inputStream = getClass.getResourceAsStream(fileName)
      readFromInputStream(inputStream)
    }
  }.toArray

  /**
    * 中心化过后的图片
    */
  lazy val pixelBytesArray: Array[Array[Array[Double]]] = {
    for (fileIndex <- 0 until 5) yield {
      val originalFileBytes = originalFileBytesArray(fileIndex)
      for (index <- 0 until 10000) yield {
        val beginIndex = index * 3073 + 1
        normalizePixel(originalFileBytes.slice(beginIndex, beginIndex + 3072))
      }
    }.toArray
  }.toArray

  /**
    * 图片对应的label
    */
  lazy val labelBytesArray: Array[Array[Int]] = {
    for (fileIndex <- 0 until 5) yield {
      val originalFileBytes = originalFileBytesArray(fileIndex)
      for (index <- 0 until 10000) yield {
        val beginIndex = index * 3073
        originalFileBytes(beginIndex).toInt
      }
    }.toArray
  }.toArray

  val random = new util.Random

  /**
    * 从inputStream中读取byte
    *
    * @param inputStream
    * @return
    */
  def readFromInputStream(inputStream: InputStream): Array[Byte] = {
    try {
      val bytes = Array.range(0, 3073 * 10000).map(_.toByte)
      inputStream.read(bytes)
      bytes
    } finally {
      inputStream.close()
    }
  }

  /**
    * 从CIFAR10文件中读图片和其对应的标签
    *
    * @param fileName CIFAR10文件名
    * @param count    要读取多少个图片和其标签
    * @return input :: expectedOutput :: HNil
    */
  def readFromResource(fileName: String,
                       count: Int): INDArray :: INDArray :: HNil = {
    val inputStream = getClass.getResourceAsStream(fileName)

    try {
      val bytes = Array.range(0, 3073 * count).map(_.toByte)
      inputStream.read(bytes)

      val labels: Seq[Double] =
        for (index <- 0 until count) yield {
          bytes(index * 3073).toDouble
        }

      val pixels: Seq[Seq[Double]] =
        for (index <- 0 until count) yield {
          for (item <- 1 until 3073) yield {
            normalizePixel(bytes(index * 3073 + item).toDouble)
          }
        }

      val labelsArray = labels.toNDArray.reshape(count, 1)
      val pixelsArray = pixels.toNDArray

      pixelsArray :: labelsArray :: HNil
    } finally {
      inputStream.close()
    }
  }

  /**
    * 归一化pixel数据
    *
    * @param pixel
    * @return
    */
  def normalizePixel(pixel: Double): Double = {
    (if (pixel < 0) {
       pixel + 256
     } else {
       pixel
     }) / 256
  }

  /**
    * 归一化数组的pixel数据
    *
    * @param original
    * @return
    */
  def normalizePixel(original: Array[Byte]): Array[Double] = {
    for (pixel <- original) yield {
      normalizePixel(pixel)
    }
  }

  /**
    * 随机获取count个train数据
    *
    * @return
    */
  def getSGDTrainNDArray(
      randomIndexArray: Array[Int]): INDArray :: INDArray :: HNil = {
    //生成0到4的随机数
    val randomIndex = random.nextInt(5)
    val labelBytes = labelBytesArray(randomIndex)
    val pixelBytes = pixelBytesArray(randomIndex)
    val count = randomIndexArray.length

    val labels: Seq[Int] =
      for (index <- 0 until count) yield {
        labelBytes(randomIndexArray(index))
      }

    val pixels: Seq[Seq[Double]] =
      for (index <- 0 until count) yield {
        pixelBytes(randomIndexArray(index)).toList
      }

    val labelsNDArray = labels.toNDArray.reshape(count, 1)
    val pixelsNDArray = pixels.toNDArray

    pixelsNDArray :: labelsNDArray :: HNil
  }
}