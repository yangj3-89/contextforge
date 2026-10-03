#!/usr/bin/env bash
# Start a throwaway PostgreSQL 16 + pgvector instance without Docker (Debian/Ubuntu).
# Requires: apt-get install postgresql-16 postgresql-16-pgvector
# Usage: sudo scripts/start_local_postgres.sh [data_dir]
set -euo pipefail

PGBIN=${PGBIN:-/usr/lib/postgresql/16/bin}
DATA_DIR=${1:-/var/lib/postgresql/contextforge}
PORT=${PGPORT:-5432}

if [ ! -d "$DATA_DIR/data" ]; then
  mkdir -p "$DATA_DIR" && chown -R postgres:postgres "$DATA_DIR"
  su postgres -c "$PGBIN/initdb -D $DATA_DIR/data -U postgres --auth=trust >/dev/null"
fi
su postgres -c "$PGBIN/pg_ctl -D $DATA_DIR/data -l $DATA_DIR/postgres.log -o '-p $PORT -k /tmp' start"
sleep 2
psql -h localhost -p "$PORT" -U postgres -tc "SELECT 1 FROM pg_roles WHERE rolname='contextforge'" | grep -q 1 || \
  psql -h localhost -p "$PORT" -U postgres -c "CREATE USER contextforge WITH PASSWORD 'contextforge' SUPERUSER;"
for db in contextforge contextforge_test; do
  psql -h localhost -p "$PORT" -U postgres -tc "SELECT 1 FROM pg_database WHERE datname='$db'" | grep -q 1 || \
    psql -h localhost -p "$PORT" -U postgres -c "CREATE DATABASE $db OWNER contextforge;"
done
echo "PostgreSQL ready: postgresql://contextforge:contextforge@localhost:$PORT/contextforge"
