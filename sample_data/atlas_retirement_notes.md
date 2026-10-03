# Retiring Project Atlas

Project Atlas was Quillon Labs' first search product, launched in 2021. It ran on a self-managed Elasticsearch cluster with keyword search only.

## Why we retired it

Atlas could not answer paraphrased questions because it had no semantic retrieval, and the Elasticsearch cluster needed frequent manual reindexing after mapping changes. Keeping Elasticsearch in sync with PostgreSQL also caused stale results after bulk imports.

## Migration

All tenants moved from Atlas to Project Halcyon between July and September 2025. Marta Kowalski wrote the migration tooling that re-ingested 41 million documents. The Elasticsearch cluster was shut down on 30 September 2025.

## Lessons

Migrations went fastest for tenants who froze document uploads for 24 hours during the cutover.
