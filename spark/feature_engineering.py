"""
spark/feature_engineering.py
Consumes the Kafka transaction stream and computes real-time velocity
features using PySpark Structured Streaming.

Trailing 5-minute velocity features per card1, computed as of each
transaction's own event time (not a clock-aligned bucket):
  - txn_count_5min
  - txn_sum_5min
  - txn_mean_5min
  - txn_stddev_5min
  - velocity_flag  (1 if count > 10 in the trailing window)

Implementation note — why this uses applyInPandasWithState instead of
a tumbling-window aggregate joined back onto the stream:
A tumbling-window groupBy + stream-stream join was tried first. It has
two problems, not one:
  1. It only closes (emits) a window after the watermark passes it,
     so a transaction's velocity features would only become available
     minutes after the transaction was actually scored — the opposite
     of "real-time" velocity detection.
  2. Re-deriving a watermarked column from a windowed aggregate's
     struct field (window.start/window.end) to satisfy the join's
     watermark requirement creates a second watermark node with
     `multipleWatermarkPolicy=min`. That second node never receives
     input until the aggregation itself emits — which it never does,
     because the global watermark is stuck at the second node's initial
     value. It's a real deadlock: confirmed by running it — the join
     validates and starts, but permanently emits zero rows.
A per-card1 stateful pass (mapGroupsWithState-style, done here as
applyInPandasWithState since PySpark's DataFrame API doesn't expose the
row-based flatMapGroupsWithState that Scala/Java get) sidesteps both:
each transaction's velocity features are computed the instant it
arrives, from that card's own trailing history, with no join and no
watermark-driven emission delay.

Known limitation: the per-card1 state (a short list of recent
timestamps/amounts, self-pruned to the last SPARK_WINDOW_DURATION) is
kept with GroupStateTimeout.NoTimeout, so it's small per key, but state
for cards that stop transacting is never freed. For a long-running
production job, switch to GroupStateTimeout.EventTimeTimeout with an
explicit state.setTimeoutTimestamp() so idle keys expire.
"""

import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, FloatType, IntegerType, LongType, DoubleType, BooleanType,
    ArrayType,
)
from pyspark.sql.streaming.state import GroupState, GroupStateTimeout

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger("feature_engineering")


def _parse_duration_seconds(duration_str: str) -> float:
    """Parse a Spark-style duration string ('5 minutes') into seconds."""
    value_str, unit = duration_str.strip().split()
    value = float(value_str)
    unit = unit.lower()
    if unit.startswith("sec"):
        return value
    if unit.startswith("min"):
        return value * 60
    if unit.startswith("hour"):
        return value * 3600
    if unit.startswith("day"):
        return value * 86400
    raise ValueError(f"Unrecognized duration unit in '{duration_str}'")


VELOCITY_WINDOW_SECONDS = _parse_duration_seconds(config.SPARK_WINDOW_DURATION)


# ─────────────────────────────────────────────
# Kafka message schema
# Full schema matching merged IEEE-CIS columns
# ─────────────────────────────────────────────
def get_transaction_schema() -> StructType:
    """Return the full Spark schema for a merged IEEE-CIS transaction row."""
    def _f(name, dtype):
        return StructField(name, dtype, nullable=True)

    fields = [
        _f("TransactionID", IntegerType()),
        _f("isFraud", IntegerType()),
        _f("TransactionDT", IntegerType()),
        _f("TransactionAmt", DoubleType()),
        _f("ProductCD", StringType()),
        _f("card1", IntegerType()),
        _f("card2", FloatType()),
        _f("card3", FloatType()),
        _f("card4", StringType()),
        _f("card5", FloatType()),
        _f("card6", StringType()),
        _f("addr1", FloatType()),
        _f("addr2", FloatType()),
        _f("dist1", FloatType()),
        _f("dist2", FloatType()),
        _f("P_emaildomain", StringType()),
        _f("R_emaildomain", StringType()),
        _f("event_time", DoubleType()),
    ]
    # C columns
    for i in range(1, 15):
        fields.append(_f(f"C{i}", FloatType()))
    # D columns
    for i in range(1, 16):
        fields.append(_f(f"D{i}", FloatType()))
    # M columns
    for i in range(1, 10):
        fields.append(_f(f"M{i}", StringType()))
    # V columns (first 15 for schema; rest are included as passthrough)
    for i in range(1, 340):
        fields.append(_f(f"V{i}", FloatType()))
    # Identity columns (from train_identity.csv merge)
    fields += [
        _f("id_01", FloatType()), _f("id_02", FloatType()),
        _f("id_03", FloatType()), _f("id_04", FloatType()),
        _f("id_05", FloatType()), _f("id_06", FloatType()),
        _f("id_09", FloatType()), _f("id_10", FloatType()),
        _f("id_11", FloatType()),
        _f("id_12", StringType()), _f("id_13", FloatType()),
        _f("id_14", FloatType()), _f("id_15", StringType()),
        _f("id_16", StringType()), _f("id_17", FloatType()),
        _f("id_18", FloatType()), _f("id_19", FloatType()),
        _f("id_20", FloatType()), _f("id_28", StringType()),
        _f("id_29", StringType()), _f("id_30", StringType()),
        _f("id_31", StringType()), _f("id_32", FloatType()),
        _f("id_33", StringType()), _f("id_34", StringType()),
        _f("id_35", StringType()), _f("id_36", StringType()),
        _f("id_37", StringType()), _f("id_38", StringType()),
        _f("DeviceType", StringType()),
        _f("DeviceInfo", StringType()),
    ]
    return StructType(fields)


def create_spark_session() -> SparkSession:
    """Create and return a configured SparkSession with Kafka support."""
    spark = (
        SparkSession.builder
        .appName(config.SPARK_APP_NAME)
        .master(config.SPARK_MASTER)
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0")
        .config("spark.sql.streaming.checkpointLocation", config.SPARK_CHECKPOINT_DIR)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    log.info("SparkSession created — master: %s", config.SPARK_MASTER)
    return spark


def read_kafka_stream(spark: SparkSession) -> DataFrame:
    """Create a streaming DataFrame from the Kafka transactions-raw topic."""
    log.info("Connecting to Kafka topic: %s", config.KAFKA_TOPIC_RAW)
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", config.KAFKA_TOPIC_RAW)
        .option("startingOffsets", config.KAFKA_AUTO_OFFSET_RESET)
        .option("maxOffsetsPerTrigger", config.SPARK_MAX_OFFSETS_PER_TRIGGER)
        .option("failOnDataLoss", "false")
        .load()
    )
    return raw_stream


def parse_transactions(raw_stream: DataFrame) -> DataFrame:
    """Parse raw Kafka JSON bytes into a typed transaction DataFrame."""
    schema = get_transaction_schema()

    parsed = (
        raw_stream
        .select(F.from_json(F.col("value").cast("string"), schema).alias("data"))
        .select("data.*")
    )

    # Convert Unix float timestamp to Spark timestamp
    parsed = parsed.withColumn(
        "event_timestamp",
        F.to_timestamp(F.col("event_time").cast("long"))
    )

    # Apply watermark for late data handling
    parsed = parsed.withWatermark("event_timestamp", config.SPARK_WATERMARK_DELAY)

    # Fill nulls in numeric columns with 0
    numeric_cols = [f.name for f in schema.fields
                    if isinstance(f.dataType, (FloatType, DoubleType, IntegerType))
                    and f.name not in ("TransactionID", "isFraud")]
    parsed = parsed.fillna(0, subset=numeric_cols)

    return parsed


def _velocity_output_schema() -> StructType:
    """
    Columns carried into the enriched output: the subset of the raw
    schema actually consumed downstream (config.GRAPH_NODE_FEATURES for
    the GNN, plus identifiers used by inference/the dashboard), the
    derived time features, and the new velocity features.
    """
    full = {f.name: f for f in get_transaction_schema().fields}
    keep_names = (
        ["TransactionID", "isFraud", "TransactionAmt", "ProductCD",
         "card1", "P_emaildomain", "event_time"]
        + config.GRAPH_NODE_FEATURES
    )
    fields = [full[name] for name in dict.fromkeys(keep_names) if name in full]
    fields += [
        StructField("log_amount", DoubleType()),
        StructField("hour_of_day", IntegerType()),
        StructField("day_of_week", IntegerType()),
        StructField("txn_count_5min", IntegerType()),
        StructField("txn_sum_5min", DoubleType()),
        StructField("txn_mean_5min", DoubleType()),
        StructField("txn_stddev_5min", DoubleType()),
        StructField("velocity_flag", IntegerType()),
    ]
    return StructType(fields)


_VELOCITY_STATE_SCHEMA = StructType([
    StructField("ts", ArrayType(DoubleType())),
    StructField("amt", ArrayType(DoubleType())),
])


def _velocity_state_func(key, pdf_iter: Iterator[pd.DataFrame], state: "GroupState"):
    """
    Per-card1 stateful pass: maintains a short trailing history of
    (event_time, amount) for this card, pruned to the last
    VELOCITY_WINDOW_SECONDS on every call, and emits each incoming
    transaction immediately with velocity features computed from that
    history — i.e. what a real-time system actually knows at the moment
    the transaction arrives, not a clock-aligned bucket average.
    """
    ts_hist, amt_hist = [], []
    if state.exists:
        stored_ts, stored_amt = state.get
        ts_hist, amt_hist = list(stored_ts), list(stored_amt)

    out_frames = []
    out_cols = [f.name for f in _velocity_output_schema()]

    for pdf in pdf_iter:
        if pdf.empty:
            continue
        pdf = pdf.sort_values("event_time")
        rows_out = []
        for _, row in pdf.iterrows():
            t = float(row["event_time"])

            cutoff = t - VELOCITY_WINDOW_SECONDS
            keep = [i for i, ts in enumerate(ts_hist) if ts > cutoff]
            ts_hist = [ts_hist[i] for i in keep]
            amt_hist = [amt_hist[i] for i in keep]

            cnt = len(ts_hist)
            s = float(sum(amt_hist))
            mean = s / cnt if cnt else 0.0
            var = (
                sum((a - mean) ** 2 for a in amt_hist) / (cnt - 1)
                if cnt > 1 else 0.0
            )
            std = var ** 0.5

            dt = datetime.fromtimestamp(t, tz=timezone.utc)
            # Spark's dayofweek() is 1=Sunday..7=Saturday; Python's
            # weekday() is 0=Monday..6=Sunday — convert to match.
            spark_dow = (dt.weekday() + 1) % 7 + 1

            out = {col: row.get(col) for col in out_cols if col in row.index}
            out["log_amount"] = math.log1p(max(float(row.get("TransactionAmt") or 0.0), 0.0))
            out["hour_of_day"] = dt.hour
            out["day_of_week"] = spark_dow
            out["txn_count_5min"] = cnt
            out["txn_sum_5min"] = round(s, 4)
            out["txn_mean_5min"] = round(mean, 4)
            out["txn_stddev_5min"] = round(std, 4)
            out["velocity_flag"] = 1 if cnt > 10 else 0
            rows_out.append(out)

            ts_hist.append(t)
            amt_hist.append(float(row.get("TransactionAmt") or 0.0))

        if rows_out:
            out_frames.append(pd.DataFrame(rows_out, columns=out_cols))

    state.update((ts_hist, amt_hist))

    if out_frames:
        yield pd.concat(out_frames, ignore_index=True)[out_cols]
    else:
        yield pd.DataFrame(columns=out_cols)


def compute_velocity_features(parsed: DataFrame) -> DataFrame:
    """
    Enrich each transaction with real-time trailing 5-minute velocity
    features per card1 (see module docstring for why this is a
    per-key stateful pass rather than a windowed aggregate + join).
    """
    return (
        parsed.groupBy("card1")
        .applyInPandasWithState(
            _velocity_state_func,
            outputStructType=_velocity_output_schema(),
            stateStructType=_VELOCITY_STATE_SCHEMA,
            outputMode="append",
            timeoutConf=GroupStateTimeout.NoTimeout,
        )
    )


def enrich_stream(parsed: DataFrame) -> DataFrame:
    """
    Add derived + velocity features to the parsed transaction stream.
    All of it — log_amount/hour_of_day/day_of_week and the velocity
    features — is computed together in compute_velocity_features()'s
    per-card1 stateful pass, since that's the single place with access
    to each transaction's own event_time.
    """
    return compute_velocity_features(parsed)


def write_to_parquet_sink(batch_df: DataFrame, batch_id: int) -> None:
    """Write each micro-batch to Parquet for downstream consumption."""
    output_path = str(config.RESULTS_DIR / "parquet" / f"batch_{batch_id}")
    batch_df.write.mode("overwrite").parquet(output_path)
    log.info("Batch %d written to Parquet — %d rows", batch_id, batch_df.count())


def start_feature_engineering_stream(spark: SparkSession, foreach_batch_fn=None):
    """
    Start the Spark Structured Streaming job.

    Args:
        spark: Active SparkSession
        foreach_batch_fn: Optional callback(batch_df, batch_id) for each micro-batch.
                          If None, defaults to writing Parquet only.
    Returns:
        StreamingQuery handle
    """
    raw = read_kafka_stream(spark)
    parsed = parse_transactions(raw)
    enriched = enrich_stream(parsed)

    handler = foreach_batch_fn if foreach_batch_fn else write_to_parquet_sink

    query = (
        enriched.writeStream
        .outputMode("append")
        .option("checkpointLocation", config.SPARK_CHECKPOINT_DIR + "/features")
        .trigger(processingTime=config.SPARK_TRIGGER_INTERVAL)
        .foreachBatch(handler)
        .start()
    )
    log.info("Feature engineering stream started — trigger: %s", config.SPARK_TRIGGER_INTERVAL)
    return query


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    spark = create_spark_session()
    query = start_feature_engineering_stream(spark)
    log.info("Streaming... press Ctrl+C to stop")
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        query.stop()
        spark.stop()
        log.info("Spark streaming stopped.")
