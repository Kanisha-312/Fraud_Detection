"""
pipeline/streaming_pipeline.py
Orchestrates the full real-time fraud detection pipeline:

  Kafka (transactions-raw)
    → Spark Structured Streaming (feature engineering)
    → GNN Inference (fraud scoring)
    → SHAP Explanation (for flagged transactions)
    → Kafka (fraud-results) + JSON file sink

Usage:
    python pipeline/streaming_pipeline.py
"""

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Configure logging early
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
log = logging.getLogger("streaming_pipeline")


def write_result_to_file(result: dict, batch_id: int) -> None:
    """Append a fraud detection result to the JSON results sink."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H")
    output_file = config.RESULTS_DIR / f"results_{ts}_batch{batch_id}.jsonl"
    with open(output_file, "a") as f:
        f.write(json.dumps(result) + "\n")


def send_result_to_kafka(producer, result: dict) -> None:
    """Send a fraud result dict to the fraud-results Kafka topic."""
    try:
        producer.send(config.KAFKA_TOPIC_RESULTS, value=result)
    except Exception as e:
        log.warning("Could not send result to Kafka: %s", e)


def process_batch(batch_df, batch_id: int, kafka_producer=None) -> None:
    """
    foreachBatch handler called by Spark for every micro-batch.

    Steps:
      1. Convert Spark DataFrame to pandas
      2. Run GNN batch inference
      3. For every fraud-flagged transaction, run SHAP explanation
      4. Write results to JSON file sink and Kafka
    """
    from inference.gnn_inference import predict_batch
    from explainability.shap_explainer import explain_transaction

    CONFIDENCE_BAND = 0.05  # matches training.train_gnn.flag_confidence's default

    t0 = time.time()
    count = batch_df.count()

    if count == 0:
        log.debug("Batch %d is empty — skipping", batch_id)
        return

    log.info("Processing batch %d — %d transactions", batch_id, count)

    # Convert to pandas
    pandas_df = batch_df.toPandas()

    # Run GNN inference — time the inference call itself so the latency
    # figure reported downstream is real per-transaction inference time,
    # not the gap between batch arrivals.
    inference_start = time.time()
    scored_df = predict_batch(pandas_df)
    inference_elapsed = time.time() - inference_start
    avg_latency_ms = (inference_elapsed / max(count, 1)) * 1000

    # Process results
    flagged_count = 0
    results = []

    for _, row in scored_df.iterrows():
        txn_id = row.get(config.TRANSACTION_ID_COL, "unknown")
        fraud_prob = float(row.get("fraud_probability", 0.0))
        fraud_flag = int(row.get("fraud_flag", 0))
        amount = float(row.get(config.AMOUNT_COL, 0.0))
        event_time = row.get("event_time", time.time())
        processed_at = datetime.utcnow().isoformat()

        explanation = None
        if fraud_flag == 1:
            flagged_count += 1
            try:
                explanation = explain_transaction(
                    transaction_id=txn_id,
                    fraud_probability=fraud_prob,
                    transaction_data=row.to_dict(),
                )
            except Exception as e:
                log.warning("Explanation failed for txn %s: %s", txn_id, e)
                explanation = {
                    "transaction_id": txn_id,
                    "fraud_probability": fraud_prob,
                    "top_features": [],
                    "suspicious_connections": [],
                    "human_readable_summary": f"Transaction flagged with {fraud_prob:.1%} probability.",
                }

        card1_val = row.get("card1")
        try:
            card1_val = int(card1_val) if card1_val is not None else None
        except (TypeError, ValueError):
            card1_val = None

        is_cold_start = bool(row.get("cold_start", False))
        near_threshold = abs(fraud_prob - config.FRAUD_THRESHOLD) <= CONFIDENCE_BAND
        # A raw probability doesn't distinguish "confidently fraud" from
        # "barely over the line" — flag both known-uncertain cases
        # (cold-start entities, and predictions near the decision
        # boundary) so they can be routed to manual review differently
        # from a confident automatic decision, instead of every score
        # being treated identically by a single threshold.
        low_confidence = is_cold_start or near_threshold

        result = {
            "transaction_id": int(txn_id) if str(txn_id).isdigit() else txn_id,
            "fraud_probability": round(fraud_prob, 4),
            "fraud_flag": fraud_flag,
            "amount": round(amount, 2),
            "card1": card1_val,
            "merchant": row.get("ProductCD"),
            "email_domain": row.get("P_emaildomain"),
            "cold_start": is_cold_start,
            "low_confidence": low_confidence,
            "event_time": float(event_time) if event_time else time.time(),
            "processed_at": processed_at,
            "latency_ms": round(avg_latency_ms, 2),
            "batch_id": batch_id,
            "explanation": explanation,
        }
        results.append(result)

        # Write to file sink
        write_result_to_file(result, batch_id)

        # Send to Kafka results topic
        if kafka_producer:
            send_result_to_kafka(kafka_producer, result)

    elapsed = time.time() - t0
    tps = count / max(elapsed, 0.001)
    log.info(
        "Batch %d done — %d txns | %d flagged | %.0f txns/sec | %.2fs",
        batch_id, count, flagged_count, tps, elapsed,
    )


def create_kafka_result_producer():
    """Create a Kafka producer for the results topic."""
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        )
        log.info("Results Kafka producer connected")
        return producer
    except Exception as e:
        log.warning("Could not create results Kafka producer: %s — results will file-only", e)
        return None


def run_pipeline() -> None:
    """Start the full Spark Structured Streaming pipeline."""
    from spark.feature_engineering import create_spark_session, read_kafka_stream, parse_transactions, enrich_stream

    log.info("=" * 60)
    log.info("Starting Fraud Detection Streaming Pipeline")
    log.info("=" * 60)

    # Initialize Kafka results producer
    kafka_producer = create_kafka_result_producer()

    # Build batch handler closure capturing kafka_producer
    def batch_handler(batch_df, batch_id):
        process_batch(batch_df, batch_id, kafka_producer)

    # Create Spark session
    spark = create_spark_session()
    log.info("Spark session ready")

    # Build streaming graph
    raw = read_kafka_stream(spark)
    parsed = parse_transactions(raw)
    enriched = enrich_stream(parsed)

    # Start streaming query
    query = (
        enriched.writeStream
        .outputMode("append")
        .option("checkpointLocation", config.SPARK_CHECKPOINT_DIR + "/pipeline")
        .trigger(processingTime=config.SPARK_TRIGGER_INTERVAL)
        .foreachBatch(batch_handler)
        .start()
    )

    log.info("Pipeline streaming started — trigger: %s", config.SPARK_TRIGGER_INTERVAL)
    log.info("Reading from Kafka topic: %s", config.KAFKA_TOPIC_RAW)
    log.info("Writing results to: %s", config.RESULTS_DIR)
    log.info("Press Ctrl+C to stop")

    # Graceful shutdown on SIGINT
    def shutdown(signum, frame):
        log.info("Shutdown signal received — stopping streaming query...")
        query.stop()
        spark.stop()
        if kafka_producer:
            kafka_producer.flush()
            kafka_producer.close()
        log.info("Pipeline shut down cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    run_pipeline()
