package MachineLearning

import java.io._
import scala.io.{BufferedSource, Source}

import scala.collection.mutable.{ListBuffer, HashMap}

/**
  * Created by raghvendra.singh on 3/1/16.
  */
class Word2Vec {
  /** This is size of the context for each training example */
  val contextSize = 5
  /** This parameter is required to chuck off those words from the vocab whose frequency is less than minFrequency */
  val minFrequency = 10
  /** This is the word's vector dimension */
  val embeddingDimension = 100
  /** This is the training text file which contains the raw text corpus */
  val trainingFile = "word2vec_training.txt"
  /** This is the maximum size of the vocab. This means maxVocabSize is the maximum number of unique words allowed in
    * the clean lemmatized training text corpus
    */
  val maxVocabSize = 30000000
  /** This is a mutable HashMap with key as a word and value as word's vector */
  val wordEmbeddings = HashMap[String, (Int,List[Double])]()
  /** This is a mutable HashMap with key as word and value as its frequency in clean lemmatized training text corpus */
  var wordFrequencyMap = HashMap[String,Int]()
  /** This is the maximum value of the standard deviation in Normal distribution */
  val maxRandVal = 0.25
  /** debugMode = 1 means do not print any statistics and debugMode = 2 means print required statistics */
  val debugMode = 1
  /** This is weight matrix between the hidden and output layer */
  var hiddenOutputEmbeddings = ListBuffer[List[Double]]()
  /** This is the count of unique words in a clean lemmatized training text corpus */
  var vocabSize = 0
  /** This is the learning rate used in gradient descent algorithm */
  val learningRate = 0.1


  def createVocabFromTrainingFile(): Unit = {
    val coreNLP = new NLP
    var bufferedSource: BufferedSource = null
    try {
      bufferedSource = Source.fromFile(trainingFile)
      for (line <- bufferedSource.getLines()) {
        /** Get all words in a sentence in lemmatized form */
        val wordList = coreNLP.tokenizeSentence(line)
        /** Get all words without any punctuation and digits in them */
        val validWordList = Utility.getValidWordList(wordList)
        validWordList foreach {
          word => {
            if (!wordFrequencyMap.contains(word)) wordFrequencyMap += ((word,1))
            else  wordFrequencyMap += ((word,wordFrequencyMap(word)+1))
          }
        }
      }
      if (minFrequency > 0) reduceVocab()
      vocabSize = wordFrequencyMap.size
      saveVocab()
    }catch {
      case ex: FileNotFoundException => println(s"Could not find file ${trainingFile}")
      case ex: IOException => println(s"Had an IOException while trying to read file ${trainingFile}")
      case ex: Exception => println("Unexpected execution error while executing method createVocabFromTrainingFile()", ex)
    } finally {
      if (bufferedSource != null) bufferedSource.close()
    }
  }

  private def reduceVocab(): Unit = {
    try {
      for (item <- wordFrequencyMap) {
        if (item._2 < minFrequency) wordFrequencyMap.remove(item._1)
      }
    } catch {
      case ex: Exception => println(s"Unexpected execution error while executing method reduceVocab()",ex)
        throw ex
    }
  }

  private def saveVocab(): Unit = {
    var fos: FileOutputStream = null
    var oos: ObjectOutputStream = null
    try {
      fos = new FileOutputStream("word_frequencies.ser")
      oos = new ObjectOutputStream(fos)
      oos.writeObject(wordFrequencyMap)
    } catch {
      case ex: Exception => println(s"Unexpected execution error while executing method saveVocab()",ex)
        throw ex
    } finally {
      if (fos != null) fos.close()
      if (oos != null) oos.close()
    }
  }

  def initializeVocabFromDisk(): Unit = {
    var fis: FileInputStream = null
    var ois: ObjectInputStream = null
    try {
      fis = new FileInputStream("word_frequencies.ser")
      ois = new ObjectInputStream(fis)
      wordFrequencyMap = ois.readObject().asInstanceOf[HashMap[String,Int]]
      vocabSize = wordFrequencyMap.size
    } catch {
      case ex: Exception => println(s"Unexpected execution error while executing method initializeVocabFromDisk()",ex)
        throw ex
    } finally {
      if (fis != null) fis.close()
      if (ois != null) ois.close()
    }
  }

  def initializeEmbeddings(): Unit = {
    try {
      /** Initialize the word embeddings of size |vocabSize| X |embeddingDimension| */
      var i = 0
      for (item <- wordFrequencyMap) {
        i = i + 1
        val lis = Utility.getRandomDoublesInRange(0, maxRandVal, embeddingDimension)
        wordEmbeddings += ((item._1, (i,lis)))
      }

      /** Initialize the hidden to output embeddings of size |vocabSize| X |embeddingDimension| */
      for (i <- 1 to vocabSize) {
        val vec = Utility.getRandomDoublesInRange(0, maxRandVal, embeddingDimension)
        hiddenOutputEmbeddings = hiddenOutputEmbeddings :+ vec
      }
    } catch {
      case ex: Exception => println(s"Unexpected execution error while executing method initializeEmbeddings()",ex)
        throw ex
    }
  }

  private def addLists(a: List[Double], b: List[Double]): List[Double] = {
    assert(a.size == b.size)
    var result = ListBuffer[Double]()
    for (i <- a.indices) result = result :+ (a(i) + b(i))
    result.toList
  }


  private def getAvgContext(context: ListBuffer[String]): List[Double] = {
    var result =  List[Double]()
    try {
      result = context.foldLeft(List.fill(embeddingDimension)(0.0)){(a,b) => addLists(a,wordEmbeddings(b)._2)}
      result
    } catch {
      case ex: Exception => println(s"Unexpected execution error while executing method initializeEmbeddings()",ex)
        throw ex
    }
  }

  private def multiply(hiddenVec: List[Double]): (List[Double],Double) = {
    assert(hiddenVec.size == hiddenOutputEmbeddings(0).size)
    var result = ListBuffer[Double]()
    var maxVal = Double.MinValue
    for (i <- hiddenOutputEmbeddings.indices) {
      var temp = 0.0
      for (j <- hiddenOutputEmbeddings(0).indices) temp = temp + (hiddenVec(j) * hiddenOutputEmbeddings(i)(j))
      result = result :+ temp
      if (maxVal < temp) maxVal = temp
    }
    (result.toList,maxVal)
  }

  private def getNormalizingFactor(maxVal: Double, output: List[Double]): Double = {
    val result = output.reduce((a,b) => math.exp(a-maxVal) + math.exp(b-maxVal))
    result
  }

  private def getSoftMaxOutput(hiddenVec: List[Double]): (List[Double], Double, Double) = {
    val result = multiply(hiddenVec)
    assert(result._1.size == vocabSize)
    val normalizingFactor = getNormalizingFactor(result._2, result._1)
    (result._1,result._2,normalizingFactor)
  }


  private def updateHiddenOutputWeight(output: (List[Double],Double,Double), targetWord: String): Unit = {
    val tempOut = output._1
    val maxVal = output._2
    val normalizingFactor = output._3
  }

  def trainWord2Vec(): Unit = {
    val coreNLP = new NLP
    var bufferedSource: BufferedSource = null
    var contextList = ListBuffer[String]()
    try {
      bufferedSource = Source.fromFile(trainingFile)
      for (line <- bufferedSource.getLines()) {
        /** Get all words in a sentence in lemmatized form */
        val wordList = coreNLP.tokenizeSentence(line)
        /** Get all words without any punctuation and digits in them */
        val validWordList = Utility.getValidWordList(wordList)
        for (word <- validWordList) {
          if(contextList.size < contextSize) {
            contextList = contextList :+ word
          } else if (contextList.size == contextSize) {
            /** Forward Phase */
            val avgContextVector = getAvgContext(contextList)
            val output = getSoftMaxOutput(avgContextVector)
            /** Backward Phase */
            updateHiddenOutputWeight(output, contextList.last)
            contextList.clear()
            contextList = contextList :+ word
          }
        }
      }
      if (minFrequency > 0) reduceVocab()
      vocabSize = wordFrequencyMap.size
      saveVocab()
    }catch {
      case ex: FileNotFoundException => println(s"Could not find file ${trainingFile}")
      case ex: IOException => println(s"Had an IOException while trying to read file ${trainingFile}")
      case ex: Exception => println("Unexpected execution error while executing method createVocabFromTrainingFile()", ex)
    } finally {
      if (bufferedSource != null) bufferedSource.close()
    }
  }


}

object ExecuteWord2Vec extends App {
  val obj = new Word2Vec
  //obj.createVocabFromTrainingFile()


}
