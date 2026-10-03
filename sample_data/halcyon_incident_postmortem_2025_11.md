# Postmortem: Halcyon indexing backlog (2025-11-18)

Incident commander: Jonas Lindqvist
Authors: Jonas Lindqvist, Alice Chen
Severity: SEV-2

## Summary

On 18 November 2025, newly uploaded documents took up to three hours to become searchable in Project Halcyon. Search over existing documents kept working. Customers saw uploads stuck in a pending state, and the ingestion API returned error QL-5031 ("indexing backlog exceeded") for 3 hours and 40 minutes.

## Root cause

A routine Kubernetes node upgrade triggered a rebalance of the Kafka consumer group used by the ingestion workers. A bug in the offset-commit logic made every worker re-process its partition from the last checkpoint after the rebalance, so consumer lag grew to roughly 2.4 million events.

## Timeline

- 09:12 Node upgrade starts and the consumer group rebalances.
- 09:40 The lag alert fires and Jonas Lindqvist declares an incident.
- 11:05 Root cause identified in the offset-commit path.
- 12:52 Patched workers deployed; the backlog drains by 13:30.

## What went well

Query traffic was unaffected because the query API reads from PostgreSQL and does not depend on Kafka. The status page was updated within 20 minutes.

## Action items

1. Commit offsets after each processed batch instead of on a timer (owner: Alice Chen).
2. Add a lag-based autoscaler for ingestion workers (owner: Jonas Lindqvist).
3. Document error QL-5031 in the customer runbook (owner: Priya Desai).
