package MachineLearning

/**
  * Created by raghvendra.singh on 3/3/16.
  */



import scala.collection.mutable.HashMap
import java.io._


class Word2Vec1 extends {

  def serialize(): Unit = {
    val hMap = HashMap("One"->List(1.1,2.2,3.3),"Two"->List(2.1,3.1,4.1),"Three"->List(9.1,8.1,7.1,6.1))
    val fos = new FileOutputStream("map.ser")
    val oos = new ObjectOutputStream(fos)
    oos.writeObject(hMap)
    oos.close()
  }

  def deSerialize(): Unit = {
    val fis = new FileInputStream("map.ser")
    val ois = new ObjectInputStream(fis)
    val anotherMap = ois.readObject().asInstanceOf[HashMap[String,List[Double]]]
    ois.close()
    println(anotherMap)
  }
}

object ExecuteWord2Vec extends  App {
  val obj = new Word2Vec1
  obj.serialize()
  obj.deSerialize()
}


