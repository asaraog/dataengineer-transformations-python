import pyspark.sql.functions as sf
from pyspark.sql import Column, DataFrame, SparkSession

METERS_PER_FOOT = 0.3048
FEET_PER_MILE = 5280
EARTH_RADIUS_IN_METERS = 6371e3
METERS_PER_MILE = METERS_PER_FOOT * FEET_PER_MILE


def haversine_distance_in_miles(
    start_latitude: Column,
    start_longitude: Column,
    end_latitude: Column,
    end_longitude: Column,
) -> Column:
    """Great-circle distance between two points given in decimal degrees."""
    start_latitude_radians = sf.radians(start_latitude.cast("double"))
    end_latitude_radians = sf.radians(end_latitude.cast("double"))
    latitude_difference = end_latitude_radians - start_latitude_radians
    longitude_difference = sf.radians(
        end_longitude.cast("double") - start_longitude.cast("double")
    )

    half_chord_squared = sf.pow(sf.sin(latitude_difference / 2), 2) + sf.cos(
        start_latitude_radians
    ) * sf.cos(end_latitude_radians) * sf.pow(sf.sin(longitude_difference / 2), 2)
    angular_distance = 2 * sf.asin(sf.sqrt(half_chord_squared))

    return angular_distance * EARTH_RADIUS_IN_METERS / METERS_PER_MILE


def compute_distance(_spark: SparkSession, dataframe: DataFrame) -> DataFrame:
    distance_in_miles = haversine_distance_in_miles(
        sf.col("start_station_latitude"),
        sf.col("start_station_longitude"),
        sf.col("end_station_latitude"),
        sf.col("end_station_longitude"),
    )
    return dataframe.withColumn("distance", sf.round(distance_in_miles, 2))


def run(
    spark: SparkSession, input_dataset_path: str, transformed_dataset_path: str
) -> None:
    input_dataset = spark.read.parquet(input_dataset_path)
    input_dataset.show()

    dataset_with_distances = compute_distance(spark, input_dataset)
    dataset_with_distances.show()

    dataset_with_distances.write.parquet(transformed_dataset_path, mode="append")
