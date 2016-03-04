name := "Word2Vec"

version := "1.0"

scalaVersion := "2.11.7"

lazy val DropOutNeuralNet_dev = (project in file(".")).
  settings(
    name := "Word2VecDevBuild"
  ).dependsOn(CoreNLP,nnutil,util)

lazy val CoreNLP = RootProject ( file("../CoreNLP") )
lazy val nnutil = RootProject ( file("../NeuralNetworkUtility") )
lazy val util = RootProject ( file("../Utility"))
    