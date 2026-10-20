"""
producer/kafka_producer.py
Streams IEEE-CIS transactions to Kafka.

Key fixes vs previous version:
  - Reads transaction CSV in chunks (low memory)
  - Merges identity ONCE into a lookup dict (not per-chunk merge)
  - Sends ALL transaction columns including dist1/dist2, C1-C14,
    D1-D15, V1-V339 — the features the GNN was trained on
  - --limit flag caps total messages for demo purposes (default 10000)
  - Logs a sample of the first message to confirm all columns present

Modes:
  fast      — fixed 50ms delay (dev/testing)
  realistic — compressed TransactionDT timing (demo)
  burst     — random batch sizes with pauses (stress test)
"""

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
log = logging.getLogger("kafka_producer")


# ── Helpers ────────────────────────────────────────────────────────────────────

def sanitize_row(row: dict) -> dict:
    """Replace NaN / inf / numpy scalars with JSON-safe Python types."""
    clean = {}
    for k, v in row.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean[k] = None
        elif isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean[k] = None if math.isnan(float(v)) else float(v)
        elif isinstance(v, (np.bool_,)):
            clean[k] = bool(v)
        else:
            clean[k] = v
    return clean


def create_producer() -> KafkaProducer:
    """Connect to Kafka with exponential-backoff retry."""
    for attempt in range(1, config.KAFKA_RETRY_ATTEMPTS + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                acks="all",
                retries=3,
                max_block_ms=15000,
            )
            log.info("Connected to Kafka at %s", config.KAFKA_BOOTSTRAP_SERVERS)
            return producer
        except NoBrokersAvailable:
            wait = (config.KAFKA_RETRY_BACKOFF_MS / 1000) * (2 ** (attempt - 1))
            log.warning("Kafka not available (attempt %d/%d). Retrying in %.1fs…",
                        attempt, config.KAFKA_RETRY_ATTEMPTS, wait)
            time.sleep(wait)
    log.error("Could not connect to Kafka after %d attempts. Is Docker running?",
              config.KAFKA_RETRY_ATTEMPTS)
    sys.exit(1)


def load_identity_lookup() -> Optional[dict]:
    """
    Load the identity CSV into a dict keyed by TransactionID.
    This is small enough (~50 MB) to hold in memory and lets us
    do O(1) identity lookups during chunked transaction streaming
    instead of re-merging every chunk.
    """
    if not config.IDENTITY_CSV.exists():
        log.warning("train_identity.csv not found — identity features will be absent")
        return None

    log.info("Loading identity CSV into lookup dict…")
    identity_df = pd.read_csv(config.IDENTITY_CSV)
    lookup = {}
    for _, row in identity_df.iterrows():
        tid = int(row[config.TRANSACTION_ID_COL])
        lookup[tid] = row.to_dict()
    log.info("Identity lookup ready — %d entries", len(lookup))
    return lookup


def enrich_row(txn_row: dict, identity_lookup: Optional[dict]) -> dict:
    """Merge identity fields into a transaction row dict."""
    if identity_lookup is None:
        return txn_row
    tid = txn_row.get(config.TRANSACTION_ID_COL)
    if tid is not None:
        try:
            id_row = identity_lookup.get(int(tid), {})
            txn_row = {**txn_row, **id_row}   # identity overwrites on collision
        except (TypeError, ValueError):
            pass
    return txn_row


def log_first_message(msg: dict) -> None:
    """Log the first message so we can verify all feature columns are present."""
    expected = set(config.GRAPH_NODE_FEATURES)
    present  = expected & set(msg.keys())
    missing  = expected - set(msg.keys())
    log.info("=" * 60)
    log.info("FIRST MESSAGE COLUMN CHECK")
    log.info("  Total columns in message : %d", len(msg))
    log.info("  Expected feature cols    : %d — %s",
             len(expected), sorted(expected))
    log.info("  Present in message       : %d — %s",
             len(present), sorted(present))
    if missing:
        log.warning("  MISSING feature cols     : %d — %s",
                    len(missing), sorted(missing))
        log.warning(
            "  The model will receive zeros for these columns. "
            "If all are missing, predictions will be constant."
        )
    else:
        log.info("  All expected feature columns are present ✓")
    log.info("  Sample values: TransactionAmt=%s  card1=%s  ProductCD=%s",
             msg.get("TransactionAmt"), msg.get("card1"), msg.get("ProductCD"))
    log.info("=" * 60)


# ── Producer modes ─────────────────────────────────────────────────────────────

def stream_transactions(
    producer: KafkaProducer,
    identity_lookup: Optional[dict],
    mode: str = "fast",
    limit: int = 10000,
) -> None:
    """
    Core streaming loop — reads transactions in 2000-row chunks,
    enriches each row with identity data, and sends to Kafka.

    Args:
        producer:         Connected KafkaProducer
        identity_lookup:  Dict of TransactionID → identity fields (or None)
        mode:             'fast' | 'realistic' | 'burst'
        limit:            Maximum number of transactions to send (0 = all)
    """
    if not config.TRANSACTION_CSV.exists():
        log.error("train_transaction.csv not found at %s", config.TRANSACTION_CSV)
        sys.exit(1)

    log.info(
        "Starting %s mode | limit: %s | Kafka: %s → topic: %s",
        mode.upper(),
        f"{limit:,}" if limit > 0 else "ALL",
        config.KAFKA_BOOTSTRAP_SERVERS,
        config.KAFKA_TOPIC_RAW,
    )

    total_sent = 0
    prev_dt    = None
    first_sent = False

    for chunk_num, chunk in enumerate(
        pd.read_csv(config.TRANSACTION_CSV, chunksize=2000)
    ):
        chunk = chunk.sort_values(config.TRANSACTION_DT_COL)

        for _, txn_row in chunk.iterrows():
            # Check limit
            if limit > 0 and total_sent >= limit:
                log.info("Reached limit of %d transactions — stopping.", limit)
                producer.flush()
                return

            # Build message dict
            msg = sanitize_row(txn_row.to_dict())
            msg = enrich_row(msg, identity_lookup)
            msg["event_time"] = time.time()

            # Log first message for diagnostics
            if not first_sent:
                log_first_message(msg)
                first_sent = True

            producer.send(config.KAFKA_TOPIC_RAW, value=msg)
            total_sent += 1

            if total_sent % 1000 == 0:
                pct = f"{100*total_sent/limit:.0f}%" if limit > 0 else ""
                log.info("Produced %d transactions %s", total_sent, pct)

            # Per-message delay
            if mode == "fast":
                time.sleep(config.PRODUCER_FAST_DELAY)

            elif mode == "realistic":
                current_dt = txn_row[config.TRANSACTION_DT_COL]
                if prev_dt is not None:
                    delay = (current_dt - prev_dt) / config.TIME_COMPRESSION_FACTOR
                    if 0 < delay < 5.0:
                        time.sleep(delay)
                prev_dt = current_dt

        # Burst: flush after each chunk + random pause
        if mode == "burst":
            producer.flush()
            pause = random.uniform(config.BURST_PAUSE_MIN, config.BURST_PAUSE_MAX)
            log.info("Burst chunk %d done | total sent: %d | pausing %.1fs",
                     chunk_num, total_sent, pause)
            time.sleep(pause)

    producer.flush()
    log.info("Streaming complete — %d total messages sent", total_sent)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IEEE-CIS Fraud Detection Kafka Producer"
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "realistic", "burst"],
        default="fast",
        help="Streaming mode (default: fast)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Max transactions to send. 0 = entire dataset (default: 10000)",
    )
    args = parser.parse_args()

    identity_lookup = load_identity_lookup()
    producer        = create_producer()

    try:
        stream_transactions(producer, identity_lookup,
                            mode=args.mode, limit=args.limit)
    except KeyboardInterrupt:
        log.info("Producer interrupted — flushing…")
        producer.flush()
        producer.close()
        log.info("Producer shut down cleanly.")


if __name__ == "__main__":
    main()