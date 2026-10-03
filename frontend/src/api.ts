// Typed client for the ContextForge API.

export const API_URL: string = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export type Mode = "lexical_only" | "vector_only" | "hybrid" | "hybrid_entity";

export interface Health {
  status: string;
  version: string;
  storage_backend: string;
  embedding_model: string;
  entity_extractor: string;
  generation_backend: string;
  counts: Record<string, number>;
}

export interface DocumentSummary {
  document_id: string;
  filename: string;
  title: string;
  source_type: string;
  created_at: string;
  chunk_count: number;
}

export interface UploadItem {
  filename: string;
  status: "indexed" | "duplicate" | "error";
  document_id: string | null;
  chunks: number;
  mentions: number;
  error: string | null;
}

export interface SearchResult {
  rank: number;
  text: string;
  source: string;
  document_id: string;
  chunk_id: string;
  title: string;
  source_type: string;
  location: Record<string, unknown>;
  retrieval_mode: string;
  vector_score: number | null;
  lexical_score: number | null;
  entity_score: number | null;
  final_score: number;
  vector_similarity: number | null;
  retrieved_by: string[];
  matched_entities: string[];
}

export interface SearchResponse {
  query: string;
  mode: Mode;
  status: "success" | "insufficient_evidence";
  confidence: number;
  message: string | null;
  results: SearchResult[];
  confidence_detail: {
    score: number;
    threshold: number;
    signals: Record<string, number>;
    reasons: string[];
    gates_triggered: string[];
  };
  weights: Record<string, number>;
  query_entities: { entity_id: string; canonical_name: string; entity_type: string; surface: string; confidence: number; method: string }[];
  unknown_entities: string[];
  candidates_considered: number;
  timings_ms: Record<string, number>;
}

export interface EntitySummary {
  entity_id: string;
  canonical_name: string;
  entity_type: string;
  aliases: string[];
  mention_count: number;
  document_count: number;
}

export interface Relationship {
  relationship_id: string;
  source_entity_id: string;
  source_name: string;
  target_entity_id: string;
  target_name: string;
  relationship_type: string;
  confidence: number;
  supporting_chunk_id: string;
  evidence: string;
  method: string;
}

export interface EntityDetail extends EntitySummary {
  mentions: { mention_id: string; surface: string; source: string | null; extractor: string; context: string }[];
  relationships: Relationship[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export const api = {
  health: () => request<Health>("/health"),
  documents: () => request<DocumentSummary[]>("/documents"),
  upload: (files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return request<{ items: UploadItem[] }>("/documents/upload", { method: "POST", body: form });
  },
  deleteDocument: (id: string) => request<void>(`/documents/${id}`, { method: "DELETE" }),
  search: (query: string, mode: Mode, topK: number) =>
    request<SearchResponse>("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, mode, top_k: topK, include_evidence_on_abstain: true }),
    }),
  entities: (type?: string, q?: string) => {
    const params = new URLSearchParams();
    if (type) params.set("entity_type", type);
    if (q) params.set("q", q);
    return request<EntitySummary[]>(`/entities?${params}`);
  },
  entity: (id: string) => request<EntityDetail>(`/entities/${id}`),
};
