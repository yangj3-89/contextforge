# Project Halcyon: Design Overview

Owner: Alice Chen (Staff Engineer, Quillon Labs)
Status: Approved, in production since October 2025

## Summary

Project Halcyon is Quillon Labs' enterprise document search platform. It replaces the legacy Project Atlas stack and lets customer analysts search contracts, tickets and internal wikis with hybrid lexical and semantic retrieval. Alice Chen leads Project Halcyon; Priya Desai owns relevance and evaluation, and Jonas Lindqvist is the SRE responsible for production reliability.

## Architecture

Documents arrive through an ingestion service that publishes one event per file to Apache Kafka. A pool of Python workers consumes those events, extracts text, splits it into passages and computes embeddings.

Halcyon stores passages, metadata and embeddings in PostgreSQL 16 with the pgvector extension, so a single database holds both the full-text index and the vector index.

The query API is a FastAPI service running on Kubernetes. Each query is embedded, matched against an HNSW index, and combined with a full-text candidate list before reranking.

## Latency and scale targets

The service level objective for interactive search is a p95 latency of 250 ms at 40 queries per second. The largest tenant today holds about 18 million passages. Index builds for a new tenant must finish within six hours.

## Reranking

The first release ranks candidates with a weighted blend of lexical and vector scores. A cross-encoder reranker is planned for the second half of 2026 once the evaluation set reaches 500 labelled queries.

## Open risks

- Kafka consumer lag during tenant onboarding can delay indexing.
- HNSW memory usage grows with tenant size, so we may need to partition indexes by tenant.
- Access control is enforced at query time; a missed filter would leak documents across tenants.
