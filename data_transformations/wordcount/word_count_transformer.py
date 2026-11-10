import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, explode, lower, split

# A word is a run of letters and apostrophes; everything else separates words.
WORD_SEPARATOR_PATTERN = r"[^\p{L}']+"


def count_words(lines: DataFrame) -> DataFrame:
    """Turn a DataFrame of text lines (column `value`) into `word`, `count` rows.

    Words are lowercased and counted exactly as written: no lemmatization or
    stemming, so "dream" and "dreams" stay separate.
    """
    words = lines.select(
        explode(split(lower(col("value")), WORD_SEPARATOR_PATTERN)).alias("word")
    ).filter(col("word") != "")

    return words.groupBy("word").count()


def run(spark: SparkSession, input_path: str, output_path: str) -> None:
    logging.info("Reading text file from: %s", input_path)
    input_df = spark.read.text(input_path)

    word_counts = count_words(input_df)

    logging.info("Writing csv to directory: %s", output_path)

    word_counts.coalesce(1).sortWithinPartitions("word").write.csv(
        output_path, header=True
    )
