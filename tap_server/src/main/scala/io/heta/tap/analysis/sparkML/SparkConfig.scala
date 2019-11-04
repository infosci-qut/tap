package io.heta.tap.analysis.sparkML

import org.apache.spark.sql.SparkSession

object SparkConfig {
  def getSparkSession(name: String): SparkSession = {
    SparkSession
      .builder()
      .appName(name)
      .master("local[3]")
      .getOrCreate()
  }
}