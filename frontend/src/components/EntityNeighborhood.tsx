import type { EntityDetail } from "../api";

// Radial view of an entity's 1-hop relationship neighborhood (strongest edge per neighbor).
export default function EntityNeighborhood({ entity, onSelect }: { entity: EntityDetail; onSelect: (id: string) => void }) {
  const best = new Map<string, { name: string; type: string; confidence: number }>();
  for (const r of entity.relationships) {
    const outgoing = r.source_entity_id === entity.entity_id;
    const id = outgoing ? r.target_entity_id : r.source_entity_id;
    const name = outgoing ? r.target_name : r.source_name;
    const prev = best.get(id);
    if (!prev || r.confidence > prev.confidence) best.set(id, { name, type: r.relationship_type, confidence: r.confidence });
  }
  const neighbors = [...best.entries()].sort((a, b) => b[1].confidence - a[1].confidence).slice(0, 12);
  if (neighbors.length === 0) return null;

  const size = 360;
  const c = size / 2;
  const radius = 130;
  return (
    <svg className="neighborhood" viewBox={`0 0 ${size} ${size}`} width={size} height={size}>
      {neighbors.map(([id, n], i) => {
        const angle = (2 * Math.PI * i) / neighbors.length - Math.PI / 2;
        const x = c + radius * Math.cos(angle);
        const y = c + radius * Math.sin(angle);
        return (
          <g key={id} onClick={() => onSelect(id)} className="nb-node">
            <line x1={c} y1={c} x2={x} y2={y} strokeWidth={1 + 3 * n.confidence} className={n.type === "related_to" ? "edge weak" : "edge"} />
            <text x={(c + x) / 2} y={(c + y) / 2 - 3} className="edge-label">
              {n.type}
            </text>
            <circle cx={x} cy={y} r={6} />
            <text x={x} y={y + 18} textAnchor="middle" className="node-label">
              {n.name}
            </text>
          </g>
        );
      })}
      <circle cx={c} cy={c} r={10} className="center" />
      <text x={c} y={c - 16} textAnchor="middle" className="node-label center-label">
        {entity.canonical_name}
      </text>
    </svg>
  );
}
