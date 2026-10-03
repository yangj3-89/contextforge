# Security review: Project Lodestar

Reviewer: Brightwater Health information security team
Date: 2026-02-03

## Data classification

Lodestar processes protected health information (PHI), including names, dates of birth, addresses and medical record numbers.

## Encryption

All PHI is encrypted at rest with AES-256. Encryption keys are stored in HashiCorp Vault and rotated every 90 days. Traffic between the Rust matching service and Kafka uses mutual TLS.

## Access control

Only the health information management team can view the manual review queue. Engineers reach production data through a break-glass procedure that requires approval from the privacy officer, and every session is logged.

## Findings

1. Medium: the DuckDB Parquet snapshots were retained for 400 days, longer than the 180-day retention policy. Remediation: automatic deletion after 180 days.
2. Low: audit logs were not forwarded to the central SIEM in real time.
