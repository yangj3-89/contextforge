# Halcyon Architecture Decision Records

## ADR-007: Use pgvector instead of a separate Elasticsearch cluster

Date: 2025-06-12. Decision makers: Alice Chen, Jonas Lindqvist.

Context: Project Atlas ran search on a self-managed Elasticsearch cluster that drifted out of sync with the primary database during bulk imports.

Decision: Halcyon keeps embeddings inside PostgreSQL using pgvector. Keeping vectors and metadata in one transactional store removes the dual-write consistency problem and takes one cluster out of the on-call rotation.

Consequences: We accept lower peak vector throughput than a dedicated vector database would give us. We will revisit this decision if any tenant exceeds 50 million passages.

## ADR-009: HNSW index parameters

Date: 2025-08-04. Decision maker: Priya Desai.

We build the HNSW index with m = 16 and ef_construction = 64 and set ef_search = 100 at query time. On the internal benchmark this reached 0.97 recall@10 against exact search while keeping the index build under four hours for 18 million passages.

## ADR-011: Serve the query API over HTTP with FastAPI

Date: 2025-09-19. Decision maker: Alice Chen.

Most customer integrations are written against REST, so the public query API is served over HTTP with FastAPI. gRPC remains in use only for internal communication between ingestion workers.

## ADR-013: Tenant isolation through row-level security

Date: 2025-10-02. Decision maker: Jonas Lindqvist.

Every passage row carries a tenant_id column, and PostgreSQL row-level security policies enforce isolation in addition to the filters applied by the application.
