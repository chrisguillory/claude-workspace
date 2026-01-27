# Project Phoenix - Technical Architecture

## Overview

Project Phoenix is a next-generation data processing platform designed to handle real-time streaming analytics at scale. The system processes over 2 million events per second with sub-millisecond latency.

## Key Components

### 1. Ingestion Layer

The ingestion layer uses Apache Kafka with a custom partitioning strategy based on tenant ID and event type. Key specifications:

- **Throughput**: 2.5M events/second sustained
- **Partitions**: 256 partitions across 32 brokers
- **Retention**: 7 days with compaction enabled
- **Replication Factor**: 3

### 2. Processing Engine

We use Apache Flink for stream processing with exactly-once semantics. The processing pipeline includes:

1. Event validation and schema enforcement
2. Enrichment with customer metadata from Redis
3. Aggregation windows (1-minute, 5-minute, 1-hour)
4. Anomaly detection using ML models

### 3. Storage Layer

#### Hot Storage (ClickHouse)
- Last 30 days of data
- Columnar format optimized for analytics
- Query latency: p99 < 100ms

#### Cold Storage (S3 + Parquet)
- Historical data beyond 30 days
- Partitioned by date and tenant
- Cost: ~$0.023 per GB/month

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/events` | POST | Ingest events |
| `/api/v1/query` | POST | Run analytics query |
| `/api/v1/health` | GET | Health check |

## Deployment

The system runs on Kubernetes with the following resource allocation:

```yaml
resources:
  requests:
    cpu: "4"
    memory: "16Gi"
  limits:
    cpu: "8"
    memory: "32Gi"
```

## Contact

- **Tech Lead**: Sarah Chen (sarah.chen@phoenix.io)
- **On-call**: #phoenix-oncall in Slack
- **Documentation**: https://docs.phoenix.io
