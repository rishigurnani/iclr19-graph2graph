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
    var same_pos = x.zip(y).count(t => t._1 == 1 & t._2 == 1)
    if (same_pos == 0) {return 0.0}
    var diff = x.zip(y).count(t => t._1 != t._2)
    var res = same_pos/(same_pos+diff).toDouble
    println(res)
    res
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkHelper.spark
    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    val fp_info = spark.read.json("data/fp_info.json") // for calculating the jaccard similarity
//    val fp_info = spark.read.json("data/vec_info.json") // for calculating the cosine similarity

    // source and target split
    val source_df = fp_info.filter($"dft_bandgap" < 4)
    val target_df = fp_info.filter($"dft_bandgap" > 6)
    // cross join, 420640 rows
    source_df.createOrReplaceTempView("source")
    target_df.createOrReplaceTempView("target")
    val df = spark.sql("select source.fp as s_fp, target.fp as t_fp, source.smiles as source_smile, target.smiles as target_smile from source CROSS JOIN target")

    // calculate the jaccard score
    def jaccardFunc(x: Seq[Double], y:Seq[Double]): Double= {
      jaccard(x, y)
    }
    val myUDF = udf(jaccardFunc _)

//    // calculate the cosine score
//    def cosineFunc(x: Seq[Double], y:Seq[Double]): Double= {
//      cosineSimilarity(x, y)
//    }
//    val myUDF = udf(cosineFunc _)

    val newDF = df.withColumn("similarity_score", myUDF(df("s_fp"), df("t_fp")))
    val res = newDF.drop("s_fp", "t_fp")

    val file_path = "fp_sim"
//    val file_path = "cosine_sim"
    res.coalesce(1).write.option("header","true").option("sep",",").mode("overwrite").csv(file_path)
    spark.stop()
  }
}
