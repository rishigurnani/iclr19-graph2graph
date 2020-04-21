package com.dan.scala

import com.dan.scala.SparkHelper // from homework
import org.apache.spark.sql.functions.udf

object CalculateSimilarity {
  def cosineSimilarity(x: Seq[Double], y: Seq[Double]): Double = {
    require(x.length == y.length)
    var res = dotProduct(x, y)/(magnitude(x) * magnitude(y))
    println(res)
    res
  }

  def dotProduct(x: Seq[Double], y: Seq[Double]): Double = {
    (for((a, b) <- x zip y) yield a * b) sum
  }

  def magnitude(x: Seq[Double]): Double = {
    math.sqrt(x map(i => i*i) sum)
  }

  def jaccard(x:Seq[Double], y: Seq[Double]): Double = {
    var samePos = x.zip(y).count(t => t._1 == 1 & t._2 == 1)
    if (samePos == 0) {return 0.0}
    var diff = x.zip(y).count(t => t._1 != t._2)
    var res = samePos/(samePos+diff).toDouble
    println(res)
    res
  }

  def main(args: Array[String]): Unit = {
    val dataFolder = args(0).toString
    val similarityMethod = args(1).toString.toUpperCase
    val demoSize = args(2).toInt
    println(dataFolder, similarityMethod, demoSize)

    // calculate the jaccard score
    def jaccardFunc(x: Seq[Double], y:Seq[Double]): Double= {
      jaccard(x, y)
    }
    // calculate the cosine score
    def cosineFunc(x: Seq[Double], y:Seq[Double]): Double= {
      cosineSimilarity(x, y)
    }

    val spark = SparkHelper.spark
    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    val info =
      if (similarityMethod == "T") {
        spark.read.json(dataFolder + "/fp_info.json")
      } else spark.read.json(dataFolder + "/vec_info.json")

    val savePath =
      if (similarityMethod == "T") {
        "fp_sim"
      } else "cosine_sim"

    val myUDF =
      if (similarityMethod == "T") {
        udf(jaccardFunc _)
    } else udf(cosineFunc _)

    info.createOrReplaceTempView("all_info")
    val dfDemo = spark.sql("select * from all_info").limit(demoSize)

    // source and target split
    val sourceDf = dfDemo.filter($"dft_bandgap" < 4)
    val targetDf = dfDemo.filter($"dft_bandgap" > 6)
    // cross join, 420640 rows
    sourceDf.createOrReplaceTempView("source")
    targetDf.createOrReplaceTempView("target")
    val df = spark.sql("select source.fp as s_fp, target.fp as t_fp, source.smiles as source_smile, target.smiles as target_smile from source CROSS JOIN target")

    val newDF = df.withColumn("similarity_score", myUDF(df("s_fp"), df("t_fp")))
    val res = newDF.drop("s_fp", "t_fp")

    // save
    res.coalesce(1).write.option("header","true").option("sep",",").mode("overwrite").csv(savePath)
    spark.stop()

  }
}
