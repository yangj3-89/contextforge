import { useEffect, useState } from "react";
import { api, type DocumentSummary, type UploadItem } from "../api";

export default function DocumentsView({ onChange }: { onChange: () => void }) {
  const [docs, setDocs] = useState<DocumentSummary[]>([]);
  const [files, setFiles] = useState<File[]>([]);
  const [results, setResults] = useState<UploadItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = () => api.documents().then(setDocs, (e: Error) => setError(e.message));
  useEffect(() => {
    void load();
  }, []);

  async function upload() {
    if (!files.length) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.upload(files);
      setResults(res.items);
      setFiles([]);
      await load();
      onChange();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function remove(id: string) {
    await api.deleteDocument(id);
    await load();
    onChange();
  }

  return (
    <section>
      <div className="upload">
        <input type="file" multiple accept=".txt,.md,.markdown,.pdf,.json" onChange={(e) => setFiles(Array.from(e.target.files ?? []))} />
        <button onClick={upload} disabled={busy || !files.length}>
          {busy ? "Indexing…" : `Upload ${files.length || ""} file${files.length === 1 ? "" : "s"}`}
        </button>
        <span className="muted small">TXT, Markdown, PDF, JSON. Entities are re-resolved after each upload.</span>
      </div>
      {error && <div className="banner error">{error}</div>}
      {results.length > 0 && (
        <ul className="upload-results">
          {results.map((r) => (
            <li key={r.filename} className={r.status}>
              {r.filename}: {r.status}
              {r.status === "indexed" && ` (${r.chunks} chunks, ${r.mentions} entity mentions)`}
              {r.error && ` — ${r.error}`}
            </li>
          ))}
        </ul>
      )}
      <table className="table">
        <thead>
          <tr>
            <th>File</th>
            <th>Title</th>
            <th>Type</th>
            <th>Chunks</th>
            <th>Indexed</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {docs.map((d) => (
            <tr key={d.document_id}>
              <td title={d.document_id}>{d.filename}</td>
              <td>{d.title}</td>
              <td>{d.source_type}</td>
              <td>{d.chunk_count}</td>
              <td>{new Date(d.created_at).toLocaleString()}</td>
              <td>
                <button className="link" onClick={() => remove(d.document_id)}>
                  delete
                </button>
              </td>
            </tr>
          ))}
          {docs.length === 0 && (
            <tr>
              <td colSpan={6} className="muted">
                No documents indexed yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </section>
  );
}
