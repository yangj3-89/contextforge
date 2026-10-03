"""Small, generic lexicons used by the rule-based extractor and resolver.

The technology lexicon doubles as a knowledge base of *known alias groups*
(e.g. Kubernetes / k8s) that pure string similarity cannot discover. Nothing in
here is specific to the bundled sample corpus beyond being common technology
names.
"""

from __future__ import annotations

# canonical name -> surface variants (matched case-sensitively, whole word).
TECHNOLOGY_ALIASES: dict[str, list[str]] = {
    "PostgreSQL": ["PostgreSQL", "Postgres", "postgres", "Postgres 16", "PostgreSQL 16"],
    "pgvector": ["pgvector"],
    "Apache Kafka": ["Apache Kafka", "Kafka"],
    "Kubernetes": ["Kubernetes", "k8s", "K8s"],
    "Docker": ["Docker"],
    "Terraform": ["Terraform"],
    "Rust": ["Rust"],
    "Python": ["Python"],
    "Go": ["Golang"],
    "TypeScript": ["TypeScript"],
    "React": ["React"],
    "FastAPI": ["FastAPI"],
    "PyTorch": ["PyTorch", "Torch"],
    "TensorFlow": ["TensorFlow"],
    "ONNX Runtime": ["ONNX Runtime", "onnxruntime", "ORT"],
    "ONNX": ["ONNX"],
    "llama.cpp": ["llama.cpp"],
    "Ollama": ["Ollama"],
    "Redis": ["Redis"],
    "DuckDB": ["DuckDB"],
    "Elasticsearch": ["Elasticsearch", "Elastic", "ES cluster"],
    "OpenSearch": ["OpenSearch"],
    "Apache Airflow": ["Apache Airflow", "Airflow"],
    "Apache Spark": ["Apache Spark", "Spark"],
    "dbt": ["dbt"],
    "Snowflake": ["Snowflake"],
    "BigQuery": ["BigQuery"],
    "Amazon S3": ["Amazon S3", "S3"],
    "AWS": ["AWS", "Amazon Web Services"],
    "GCP": ["GCP", "Google Cloud"],
    "gRPC": ["gRPC"],
    "GraphQL": ["GraphQL"],
    "Grafana": ["Grafana"],
    "Prometheus": ["Prometheus"],
    "OpenTelemetry": ["OpenTelemetry", "OTel"],
    "OR-Tools": ["OR-Tools", "Google OR-Tools"],
    "spaCy": ["spaCy"],
    "sentence-transformers": ["sentence-transformers", "SentenceTransformers"],
    "MiniLM": ["MiniLM", "all-MiniLM-L6-v2"],
    "HNSW": ["HNSW"],
    "BM25": ["BM25"],
    "CUDA": ["CUDA"],
    "TensorRT": ["TensorRT"],
    "Core ML": ["Core ML", "CoreML"],
    "WebAssembly": ["WebAssembly", "WASM"],
    "SQLite": ["SQLite"],
    "MongoDB": ["MongoDB", "Mongo"],
    "ClickHouse": ["ClickHouse"],
    "Flink": ["Apache Flink", "Flink"],
    "Temporal": ["Temporal.io"],
    "Splink": ["Splink"],
    "HL7 FHIR": ["HL7 FHIR", "FHIR"],
    "AES-256": ["AES-256"],
    "Android": ["Android"],
    "iOS": ["iOS"],
    "Apache Parquet": ["Apache Parquet", "Parquet"],
    "Vault": ["HashiCorp Vault"],
}

# surface variant -> canonical (used for the known-alias resolution feature).
TECHNOLOGY_CANONICAL: dict[str, str] = {
    variant.lower(): canonical
    for canonical, variants in TECHNOLOGY_ALIASES.items()
    for variant in variants
}

ORG_SUFFIXES: frozenset[str] = frozenset(
    {
        "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited", "co",
        "company", "plc", "gmbh", "sa", "ag", "bv", "pte",
    }
)

# Words that signal an organization name when they end a capitalized sequence.
ORG_HEAD_WORDS: tuple[str, ...] = (
    "Inc", "Inc.", "Labs", "Lab", "Corp", "Corp.", "Corporation", "LLC", "Ltd", "Ltd.",
    "Logistics", "Health", "Systems", "Analytics", "Technologies", "University",
    "Capital", "Data", "AI", "Robotics", "Bank", "Group", "Partners", "Foundation",
    "Institute", "Hospital", "Networks", "Software", "Ventures",
)

PERSON_TITLES: frozenset[str] = frozenset({"dr", "mr", "mrs", "ms", "prof", "professor", "sir"})

PROJECT_WORDS: frozenset[str] = frozenset({"project", "initiative", "program", "programme", "pilot"})

# A handful of large cities as a fallback location lexicon when spaCy is unavailable.
CITIES: tuple[str, ...] = (
    "New York", "San Francisco", "Seattle", "Austin", "Boston", "Chicago", "Toronto",
    "Montreal", "Vancouver", "London", "Paris", "Berlin", "Munich", "Amsterdam", "Lisbon",
    "Madrid", "Barcelona", "Dublin", "Zurich", "Stockholm", "Warsaw", "Singapore", "Tokyo",
    "Seoul", "Bangalore", "Bengaluru", "Mumbai", "Sydney", "Melbourne", "Lagos", "Nairobi",
    "Sao Paulo", "Mexico City", "Denver", "Atlanta", "Rotterdam", "Hamburg", "Oslo",
    "Porto", "Lyon", "Milan", "Vienna", "Prague", "Copenhagen", "Helsinki",
)

# Common technical/business acronyms that statistical NER frequently mislabels as organizations.
ACRONYM_STOPLIST: frozenset[str] = frozenset(
    """
    API APIS GPU GPUS CPU CPUS TPU RAM GB MB TB KB SLA SLO SLI SRE TLS SSL PDF CSV JSON XML HTML HTTP
    HTTPS URL URI SQL EHR EMR PHI PII HIPAA GDPR SOC ML AI LLM NLP UI UX QA CI CD PR KPI OKR HR IT ID
    SDK ORM VM OS SSO MFA RBAC SIEM REST RPC CEO CTO CFO COO VP PM EM IC SEV ETA FAQ MVP POC RFC ADR
    """.split()
)

# Capitalized tokens that should never be treated as a project / org head.
NON_NAME_WORDS: frozenset[str] = frozenset(
    {
        "The", "This", "That", "These", "Our", "Their", "A", "An", "Pilot", "Project", "New",
        "Next", "Phase", "Q1", "Q2", "Q3", "Q4", "Each", "Every", "One", "Two", "It", "We",
        "Data", "Health", "Labs", "Systems", "Status", "Summary", "Notes", "Team", "Overview",
        "January", "February", "March", "April", "May", "June", "July", "August", "September",
        "October", "November", "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        "Saturday", "Sunday", "Today", "Weekly", "Daily", "Quarterly", "Annual",
    }
)
