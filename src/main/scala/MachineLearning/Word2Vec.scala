package MachineLearning

import java.io._
import scala.io.{BufferedSource, Source}
import System.nanoTime
import scala.collection.mutable.{ListBuffer, HashMap}

/**
  * Created by raghvendra.singh on 3/1/16.
  */

case class HuffmanNode(var outputWordVec: List[Double], left: HuffmanNode, right: HuffmanNode, word: String)

class Word2Vec {
  /** This is size of the context for each training example */
  val contextSize = 5
  /** This parameter is required to chuck off those words from the vocab whose frequency is less than minFrequency */
  val minFrequency = 5
  /** This is the word's vector dimension */
  val embeddingDimension = 100
  /** This is the training text file which contains the raw text corpus */
  val trainingFile = "word2vec_training.txt"
  /** This is the maximum size of the vocab. This means maxVocabSize is the maximum number of unique words allowed in
    * the clean training text corpus
    */
  val maxVocabSize = 30000000
  /** This is a mutable HashMap with key as a word and value as word's vector */
  val wordEmbeddings = HashMap[String, (Int,List[Double])]()
  /** This is a mutable HashMap with key as word and value as its frequency in clean training text corpus */
  var wordFrequencyMap = HashMap[String,(Int,String)]()
  /** This is the maximum value of the standard deviation in Normal distribution */
  val maxRandVal = 0.25
  /** debugMode = 0 means do not print any statistics and debugMode = 1 means print required statistics */
  val debugMode = 0
  /** This is weight matrix between the hidden and output layer */
  var hiddenOutputEmbeddings = ListBuffer[List[Double]]()
  /** This is the count of unique words in a clean training text corpus */
  var vocabSize = 0
  /** This is the learning rate used in gradient descent algorithm */
  val learningRate = 0.025
  /** This is the root node of the binary Huffman tree */
  var rootHuffman: HuffmanNode = null
  /** This is a flag to choose hierarchical softmax vs general softmax */
  var isHierarchicalSoftmax = 1



  def createVocabFromTrainingFile(): Unit = {
    println("Creating word frequency map from training corpus.")
    val coreNLP = new NLP
    var bufferedSource: BufferedSource = null
    try {
      bufferedSource = Source.fromFile(trainingFile)
      for (line <- bufferedSource.getLines()) {
        /** Get all words in a sentence in form */
        val wordList = coreNLP.tokenizeParagraph(line)
        /** Get all words without any punctuation and digits in them */
        val validWordList = Utility.getValidWordList(wordList)
        validWordList foreach {
          word => {
            if (!wordFrequencyMap.contains(word)) wordFrequencyMap += ((word,(1,"")))
            else  wordFrequencyMap += (  (  word,(wordFrequencyMap(word)._1+1,"")  )  )
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
        if (item._2._1 < minFrequency) wordFrequencyMap.remove(item._1)
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
      wordFrequencyMap = ois.readObject().asInstanceOf[HashMap[String,(Int,String)]]
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


  private def getAvgContext(context: ListBuffer[String]): List[Double] = {
    var result =  List[Double]()
    /** get actual context by removing the target word */
    val actualContext = context.slice(0,context.size-1)
    try {
      result = actualContext.foldLeft(List.fill(embeddingDimension)(0.0)){(a,b) => Utility.addLists(a,wordEmbeddings(b)._2)} map (x => x/contextSize)
      result
    } catch {
      case ex: Exception => println(s"Unexpected execution error while executing method getAvgContext()",ex)
        throw ex
    }
  }

  private def buildBinaryHuffmanTree(): Unit = {
    println("Started Building Binary Huffman Tree")
    var oldList = ListBuffer[HuffmanNode]()
    for (item <- wordFrequencyMap) oldList += HuffmanNode(List[Double](), null, null, item._1)
    while(oldList.size != 1) {
      var newList = ListBuffer[HuffmanNode]()
      var i = 0
      while (i < oldList.size - 1) {
        newList += HuffmanNode(Utility.getRandomDoublesInRange(0, maxRandVal, embeddingDimension), oldList(i), oldList(i + 1), "")
        i = i + 2
      }
      if (i == oldList.size - 1) newList += oldList(i)
      oldList = newList
    }
    if (oldList.size == 1) rootHuffman = oldList.head
  }


  private def assignHuffmanCodeToWordsUtil(root: HuffmanNode, code: String): Unit = {
    if (root != null) {
      if (root.left == null && root.right == null) {
        val value = wordFrequencyMap(root.word)
        wordFrequencyMap(root.word) = (value._1, code)
        return
      }
      assignHuffmanCodeToWordsUtil(root.left, code + "0")
      assignHuffmanCodeToWordsUtil(root.right, code + "1")
    }
  }

  private def assignHuffmanCodeToWords(): Unit = {
    assignHuffmanCodeToWordsUtil(rootHuffman, "")
  }



  private def multiply(hiddenVec: List[Double]): (List[Double],Double) = {
    assert(hiddenVec.size == hiddenOutputEmbeddings(0).size)
    var result = ListBuffer[Double]()
    var maxVal = Double.MinValue
    try {
      for (i <- hiddenOutputEmbeddings.indices) {
        var temp = 0.0
        for (j <- hiddenOutputEmbeddings(0).indices) temp = temp + (hiddenVec(j) * hiddenOutputEmbeddings(i)(j))
        result = result :+ temp
        if (maxVal < temp) maxVal = temp
      }
      (result.toList, maxVal)
    } catch {
      case ex: Exception => println(s"Unexpected execution error while executing method multiply()",ex)
        throw ex
    }
  }

  private def getNormalizingFactor(maxVal: Double, output: List[Double]): Double = {
    var result = 0.0
    try {
      for (i <- output.indices) result = result + math.exp(output(i) - maxVal)
      result
    } catch {
      case ex: Exception => println(s"Unexpected execution error while executing method getNormalizingFactor()",ex)
        throw ex
    }
  }

  private def getSoftMaxOutput(hiddenVec: List[Double]): (List[Double], Double, Double) = {
    val result = multiply(hiddenVec)
    assert(result._1.size == vocabSize)
    try {
      val normalizingFactor = getNormalizingFactor(result._2, result._1)
      (result._1, result._2, normalizingFactor)
    } catch {
      case ex: Exception => println(s"Unexpected execution error while executing method getSoftMaxOutput()",ex)
        throw ex
    }
  }


  private def updateHiddenOutputWeigtsHierarchicalSoftmax(hiddenVec: List[Double],  targetWord: String): List[Double] = {
    var root = rootHuffman
    var deltaHidden = List.fill(hiddenVec.size)(0.0)
    for (char <- wordFrequencyMap(targetWord)._2) {
      var newRoot: HuffmanNode = null
      val temp = Utility.dotProductLists(root.outputWordVec, hiddenVec)
      var temp1 = 1.0/(1 + math.exp(temp))
      if (char == '0') {
        temp1 = temp1 - 1.0
        newRoot = root.left
      } else newRoot = root.right
      val temp2 = Utility.multiplyScalarToList(hiddenVec, learningRate * temp1)
      val temp3 = Utility.multiplyScalarToList(root.outputWordVec, temp1)
      deltaHidden = Utility.addLists(temp3, deltaHidden)
      root.outputWordVec = Utility.subtractLists(root.outputWordVec, temp2)
      root = newRoot
    }
    deltaHidden
  }


  private def updateHiddenOutputWeights(output: (List[Double],Double,Double), hiddenVec: List[Double], targetWord: String): Unit = {
    val tempOut = output._1
    val maxVal = output._2
    val normalizingFactor = output._3
    try {
      for (i <- hiddenOutputEmbeddings.indices) {
        var delta = math.exp(tempOut(i) - maxVal) / normalizingFactor
        if (wordEmbeddings(targetWord)._1 - 1 == i)
          delta = delta - 1
        val newVec = Utility.subtractLists(hiddenOutputEmbeddings(i), Utility.multiplyScalarToList(hiddenVec, learningRate * delta))
        hiddenOutputEmbeddings(i) = newVec
      }
    } catch {
      case ex: Exception => println(s"Unexpected execution error while executing method updateHiddenOutputWeights()",ex)
        throw ex
    }
  }

  private def getDeltaHidden(output: (List[Double],Double,Double)): List[Double] = {
    var result = ListBuffer[Double]()
    val tempOut = output._1
    val maxVal = output._2
    val normalizingFactor = output._3
    try {
      for (i <- hiddenOutputEmbeddings(0).indices) {
        var mul = 0.0
        for (j <- hiddenOutputEmbeddings.indices) mul = mul + (math.exp(tempOut(j) - maxVal) / normalizingFactor) * hiddenOutputEmbeddings(j)(i)
        result = result :+ mul
      }
      assert(result.size == embeddingDimension)
      result.toList
    }  catch {
      case ex: Exception => println(s"Unexpected execution error while executing method getDeltaHidden()",ex)
        throw ex
    }
  }

  private def updateInputHiddenWeights(deltaHidden: List[Double], contextList: ListBuffer[String]): Unit = {
    val numContexts = contextList.size-1
    assert(contextSize == numContexts)
    try {
      for (i <- 0 to numContexts - 1) {
        val tup = wordEmbeddings(contextList(i))
        val update = Utility.multiplyScalarToList(deltaHidden, learningRate / contextSize)
        wordEmbeddings(contextList(i)) = (tup._1, Utility.subtractLists(tup._2, update))
      }
    } catch {
      case ex: Exception => println(s"Unexpected execution error while executing method updateInputHiddenWeights()",ex)
        throw ex
    }
  }

  def trainWord2Vec(): Unit = {
    val coreNLP = new NLP
    var bufferedSource: BufferedSource = null
    var contextList = ListBuffer[String]()
    try {
      bufferedSource = Source.fromFile(trainingFile)
      for (line <- bufferedSource.getLines()) {
        /** Get all words in a sentence in form */
        val wordList = coreNLP.tokenizeParagraph(line)
        /** Get all words without any punctuation and digits in them */
        val validWordList = Utility.getValidWordList(wordList)
        for (word <- validWordList if wordFrequencyMap.contains(word)) {
          /** Prepare a context list having last word as target word */
          if(contextList.size < contextSize+1) {
            contextList = contextList :+ word
          } else if (contextList.size == contextSize+1) {
            if (debugMode == 1) {
              println()
              println("ContextList is=")
              println()
              println(contextList)
              println("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            }
            /** Forward Phase */
            val avgContextVector = getAvgContext(contextList)
            if (debugMode == 1) {
              println()
              println("Avg. Context Vector is=")
              println()
              println(avgContextVector)
              println("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            }

            val output = getSoftMaxOutput(avgContextVector)
            if (debugMode == 1) {
              println()
              println("softmax output is=")
              println()
              println(output)
              println("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            }
            /** Backward Phase */
            val deltaHidden = getDeltaHidden(output)
            if (debugMode == 1) {
              println()
              println("delta hidden is=")
              println()
              println(deltaHidden)
              println("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            }
            updateHiddenOutputWeights(output, avgContextVector, contextList.last)
            if (debugMode == 1) {
              println()
              println("hidden output embeddings is=")
              println()
              println(hiddenOutputEmbeddings)
              println("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            }
            updateInputHiddenWeights(deltaHidden, contextList)
            if (debugMode == 1) {
              println()
              println("word embeddings is=")
              println()
              println(wordEmbeddings)
              println("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            }
            contextList.remove(0)
            contextList = contextList :+ word
            if (debugMode == 1) println("**************************__________________*******************__________________*****************")
          }
        }
      }
    }catch {
      case ex: FileNotFoundException => println(s"Could not find file ${trainingFile}")
      case ex: IOException => println(s"Had an IOException while trying to read file ${trainingFile}")
      case ex: Exception => println("Unexpected execution error while executing method trainWord2Vec()", ex)
    } finally {
      if (bufferedSource != null) bufferedSource.close()
    }
  }


  def trainWord2VecHierarchicalSoftmax(): Unit = {
    val coreNLP = new NLP
    var bufferedSource: BufferedSource = null
    var contextList = ListBuffer[String]()
    try {
      /** Build a Binary Huffman Tree */
      buildBinaryHuffmanTree()
      /** Assign a code to each word in wordFrequencyMap by traversing the Binary Huffman Tree */
      assignHuffmanCodeToWords()
      bufferedSource = Source.fromFile(trainingFile)
      for (line <- bufferedSource.getLines()) {
        /** Get all words in a sentence in form */
        val wordList = coreNLP.tokenizeParagraph(line)
        /** Get all words without any punctuation and digits in them */
        val validWordList = Utility.getValidWordList(wordList)
        for (word <- validWordList if wordFrequencyMap.contains(word)) {
          /** Prepare a context list having last word as target word */
          if(contextList.size < contextSize+1) {
            contextList = contextList :+ word
          } else if (contextList.size == contextSize+1) {
            if (debugMode == 1) {
              println()
              println("ContextList is=")
              println()
              println(contextList)
              println("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            }
            /** Forward Phase */
            val avgContextVector = getAvgContext(contextList)
            if (debugMode == 1) {
              println()
              println("Avg. Context Vector is=")
              println()
              println(avgContextVector)
              println("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            }
            /** Backward Phase */

            val deltaHidden = updateHiddenOutputWeigtsHierarchicalSoftmax(avgContextVector, contextList.last)
            updateInputHiddenWeights(deltaHidden, contextList)
            if (debugMode == 1) {
              println()
              println("word embeddings is=")
              println()
              println(wordEmbeddings)
              println("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            }
            contextList.remove(0)
            contextList = contextList :+ word
            if (debugMode == 1) println("**************************__________________*******************__________________*****************")
          }
        }
      }
    }catch {
      case ex: FileNotFoundException => println(s"Could not find file ${trainingFile}")
      case ex: IOException => println(s"Had an IOException while trying to read file ${trainingFile}")
      case ex: Exception => println("Unexpected execution error while executing method trainWord2Vec()", ex)
    } finally {
      if (bufferedSource != null) bufferedSource.close()
    }
  }

  def saveWordVectors(): Unit = {
    var fos: FileOutputStream = null
    var oos: ObjectOutputStream = null
    try {
      fos = new FileOutputStream("word_vectors.ser")
      oos = new ObjectOutputStream(fos)
      oos.writeObject(wordEmbeddings)
    } catch {
      case ex: Exception => println(s"Unexpected execution error while executing method saveWordVectors()",ex)
        throw ex
    } finally {
      if (fos != null) fos.close()
      if (oos != null) oos.close()
    }
  }

}

object ExecuteWord2Vec extends App {
  val obj = new Word2Vec
  var startTime = 0L
  var endTime = 0L
  if (new java.io.File("word_frequencies.ser").exists) {
    startTime = System.nanoTime()
    obj.initializeVocabFromDisk()
    endTime = System.nanoTime()
    println("Successfully initialized word frequency map from word frequency map stored in disk.")
    println("Time taken to initialize word frequency map from word frequency map stored in disk = "+ (endTime-startTime)/(math.pow(10,9)*60) + "minutes")
  }
  else {
    startTime = System.nanoTime()
    obj.createVocabFromTrainingFile()
    endTime = System.nanoTime()
    println("Successfully created word frequency map from training corpus.")
    println("Time taken to create word frequency map from training corpus = "+ (endTime-startTime)/(math.pow(10,9)*60) + "minutes")
  }

  if(obj.debugMode == 1) {
    println("WordFrequencyMap is=")
    println()
    println(obj.wordFrequencyMap)
    println("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    println()
    println("vocab size is = "+obj.vocabSize)
    println("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
  }

  obj.initializeEmbeddings()
  if(obj.debugMode == 1) {
    println("wordEmbeddings is=")
    println()
    println(obj.wordEmbeddings)
    println("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    println()
    println("HiddenOutputEmbeddings is=")
    println()
    println(obj.hiddenOutputEmbeddings)
    println("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
  }
  println("Started training Word2Vec....")
  val startTrainingTime = System.nanoTime()
  if (obj.isHierarchicalSoftmax == 1) obj.trainWord2VecHierarchicalSoftmax()
  else obj.trainWord2Vec()
  val endTrainingTime = System.nanoTime()
  println("Time taken to train Word2Vec = "+ (endTrainingTime-startTrainingTime)/(math.pow(10,9)*60) + "minutes")
  println("Successfully trained Word2Vec model.")

  println("Started saving word vectors to disk...")
  obj.saveWordVectors()
  println("Successfully saved word vectors to disk.")
  /*val coreNLP = new NLP
  val wordList = coreNLP.tokenizeParagraph("rama is going to market by his bicycle.")
  println("word list is = "+wordList)
  val validWordList = Utility.getValidWordList(wordList)
  println("valid word list is = "+validWordList)*/
}
