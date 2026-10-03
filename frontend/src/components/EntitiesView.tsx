import { useEffect, useState } from "react";
import { api, type EntityDetail, type EntitySummary } from "../api";
import EntityNeighborhood from "./EntityNeighborhood";

const TYPES = ["", "PERSON", "ORGANIZATION", "PROJECT", "TECHNOLOGY", "LOCATION", "EVENT", "DATE"];

export default function EntitiesView() {
  const [type, setType] = useState("");
  const [q, setQ] = useState("");
  const [entities, setEntities] = useState<EntitySummary[]>([]);
  const [selected, setSelected] = useState<EntityDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let current = true; // ignore responses from superseded requests
    api.entities(type || undefined, q || undefined).then(
      (list) => current && setEntities(list),
      (e: Error) => current && setError(e.message),
    );
    return () => {
      current = false;
    };
  }, [type, q]);

  const open = (id: string) => api.entity(id).then(setSelected, (e: Error) => setError(e.message));

  return (
    <section className="entities">
      <div className="entity-list">
        <div className="filters">
          <select value={type} onChange={(e) => setType(e.target.value)}>
            {TYPES.map((t) => (
              <option key={t} value={t}>
                {t || "All types"}
              </option>
            ))}
          </select>
          <input placeholder="filter by name or alias" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        {error && <div className="banner error">{error}</div>}
        <ul>
          {entities.map((e) => (
            <li key={e.entity_id} className={selected?.entity_id === e.entity_id ? "selected" : ""} onClick={() => open(e.entity_id)}>
              <span className={`entity-tag t-${e.entity_type}`}>{e.entity_type.slice(0, 4)}</span> {e.canonical_name}
              <span className="muted small"> · {e.mention_count} mentions</span>
              {e.aliases.length > 0 && <div className="muted small">aka {e.aliases.join(", ")}</div>}
            </li>
          ))}
        </ul>
      </div>
      <div className="entity-detail">
        {!selected && <p className="muted">Select an entity to see its aliases, relationships (with evidence) and mentions.</p>}
        {selected && (
          <>
            <h2>
              {selected.canonical_name} <span className={`entity-tag t-${selected.entity_type}`}>{selected.entity_type}</span>
            </h2>
            <p className="muted">
              {selected.mention_count} mentions in {selected.document_count} documents
              {selected.aliases.length > 0 && <> · resolved aliases: {selected.aliases.join(", ")}</>}
            </p>
            <EntityNeighborhood entity={selected} onSelect={open} />
            <h3>Relationships</h3>
            <table className="table small">
              <thead>
                <tr>
                  <th>Relation</th>
                  <th>Confidence</th>
                  <th>Method</th>
                  <th>Evidence</th>
                </tr>
              </thead>
              <tbody>
                {selected.relationships.map((r) => (
                  <tr key={r.relationship_id}>
                    <td>
                      {r.source_name} <b>{r.relationship_type}</b> {r.target_name}
                    </td>
                    <td>{r.confidence.toFixed(2)}</td>
                    <td>{r.method}</td>
                    <td className="evidence" title={r.supporting_chunk_id}>
                      {r.evidence}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <h3>Mentions</h3>
            <ul className="mentions">
              {selected.mentions.map((m) => (
                <li key={m.mention_id}>
                  <b>{m.surface}</b> <span className="muted small">({m.source}, {m.extractor})</span>
                  <div className="small">{m.context}</div>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </section>
  );
}
