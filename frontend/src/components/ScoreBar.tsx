interface Props {
  label: string;
  value: number | null;
  kind: "vector" | "lexical" | "entity" | "final";
}

export default function ScoreBar({ label, value, kind }: Props) {
  return (
    <div className="score">
      <span className="score-label">{label}</span>
      <div className="score-track">
        {value !== null && <div className={`score-fill ${kind}`} style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />}
      </div>
      <span className="score-value">{value === null ? "n/a" : value.toFixed(3)}</span>
    </div>
  );
}
