import { useState, type FormEvent } from "react";
import { api, type Mode, type SearchResponse } from "../api";
import ScoreBar from "./ScoreBar";

const MODES: { value: Mode; label: string }[] = [
  { value: "lexical_only", label: "Lexical (BM25)" },
  { value: "vector_only", label: "Vector (MiniLM)" },
  { value: "hybrid", label: "Hybrid" },
  { value: "hybrid_entity", label: "Hybrid + Entity" },
];

const EXAMPLES = [
  "Which database does the project led by Alice Chen use?",
  "What does error QL-5031 mean?",
  "How do we stop one customer from seeing another customer's documents?",
  "What did Bob Smith present at the Quillon Offsite?",
];

function location(loc: Record<string, unknown>): string {
  const parts: string[] = [];
  if (loc.section) parts.push(`§ ${String(loc.section)}`);
  if (loc.page_start) parts.push(`p. ${String(loc.page_start)}${loc.page_end !== loc.page_start ? `–${String(loc.page_end)}` : ""}`);
  if (loc.json_path) parts.push(String(loc.json_path));
  return parts.join(" · ");
}

export default function SearchView() {
  const [query, setQuery] = useState(EXAMPLES[0]);
  const [mode, setMode] = useState<Mode>("hybrid_entity");
  const [topK, setTopK] = useState(5);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function run(e?: FormEvent, q: string = query) {
    e?.preventDefault();
    if (!q.trim()) return;
    setLoading(true);
    setError(null);
    try {
      setResponse(await api.search(q, mode, topK));
    } catch (err) {
      setError((err as Error).message);
      setResponse(null);
    } finally {
      setLoading(false);
    }
  }

  const abstained = response?.status === "insufficient_evidence";

  return (
    <section>
      <form className="search-form" onSubmit={run}>
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Ask a question about the indexed documents" />
        <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
          {MODES.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
        <select value={topK} onChange={(e) => setTopK(Number(e.target.value))} title="top k">
          {[3, 5, 10].map((k) => (
            <option key={k} value={k}>
              top {k}
            </option>
          ))}
        </select>
        <button type="submit" disabled={loading}>
          {loading ? "Searching…" : "Search"}
        </button>
      </form>
      <div className="examples">
        {EXAMPLES.map((ex) => (
          <button key={ex} className="chip" onClick={() => { setQuery(ex); void run(undefined, ex); }}>
            {ex}
          </button>
        ))}
      </div>

      {error && <div className="banner error">{error}</div>}

      {response && (
        <>
          <div className={abstained ? "banner abstain" : "banner success"}>
            <div className="banner-title">
              {abstained ? "Insufficient evidence — ContextForge abstained" : "Evidence found"}
              <span className="confidence">
                confidence {response.confidence.toFixed(2)} (threshold {response.confidence_detail.threshold.toFixed(2)})
              </span>
            </div>
            {abstained && <p>{response.message}</p>}
            {response.confidence_detail.reasons.length > 0 && (
              <ul className="reasons">
                {response.confidence_detail.reasons.map((r) => (
                  <li key={r}>{r}</li>
                ))}
              </ul>
            )}
            <div className="signals">
              {Object.entries(response.confidence_detail.signals).map(([k, v]) => (
                <span key={k} className="signal">
                  {k}: <b>{v.toFixed(2)}</b>
                </span>
              ))}
            </div>
          </div>

          <div className="meta-row">
            {response.query_entities.length > 0 && (
              <span>
                Linked entities:{" "}
                {response.query_entities.map((e) => (
                  <span key={e.entity_id + e.surface} className={`entity-tag t-${e.entity_type}`} title={`${e.method}, conf ${e.confidence}`}>
                    {e.canonical_name}
                  </span>
                ))}
              </span>
            )}
            {response.unknown_entities.length > 0 && <span className="warn">Unknown: {response.unknown_entities.join(", ")}</span>}
            <span>
              weights α={response.weights.alpha_vector} β={response.weights.beta_lexical} γ={response.weights.gamma_entity}
            </span>
            <span>
              {response.candidates_considered} candidates · {response.timings_ms.total_ms?.toFixed(1)} ms
            </span>
          </div>

          {abstained && response.results.length > 0 && (
            <p className="muted">Rejected candidates (shown for debugging, not returned as an answer):</p>
          )}
          <div className={abstained ? "results rejected" : "results"}>
            {response.results.map((r) => (
              <article key={r.chunk_id} className="card">
                <header>
                  <span className="rank">#{r.rank}</span>
                  <span className="source">{r.source}</span>
                  <span className="muted">{location(r.location)}</span>
                  <span className="badges">
                    {r.retrieved_by.map((s) => (
                      <span key={s} className={`badge ${s}`}>
                        {s}
                      </span>
                    ))}
                  </span>
                </header>
                <pre className="passage">{r.text}</pre>
                {r.matched_entities.length > 0 && <div className="muted small">Entities: {r.matched_entities.join(", ")}</div>}
                <div className="scores">
                  <ScoreBar label="vector" value={r.vector_score} kind="vector" />
                  <ScoreBar label="lexical" value={r.lexical_score} kind="lexical" />
                  <ScoreBar label="entity" value={r.entity_score} kind="entity" />
                  <ScoreBar label="final" value={r.final_score} kind="final" />
                </div>
                <footer className="muted small">
                  chunk {r.chunk_id}
                  {r.vector_similarity !== null && ` · cosine ${r.vector_similarity.toFixed(3)}`}
                </footer>
              </article>
            ))}
          </div>
        </>
      )}
    </section>
  );
}
