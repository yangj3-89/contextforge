import { useCallback, useEffect, useState } from "react";
import { api, API_URL, type Health } from "./api";
import SearchView from "./components/SearchView";
import DocumentsView from "./components/DocumentsView";
import EntitiesView from "./components/EntitiesView";

type Tab = "search" | "documents" | "entities";

export default function App() {
  const [tab, setTab] = useState<Tab>("search");
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  const refreshHealth = useCallback(() => {
    api.health().then(setHealth, (e: Error) => setHealthError(e.message));
  }, []);
  useEffect(refreshHealth, [refreshHealth]);

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>ContextForge</h1>
          <p className="subtitle">Provenance-aware hybrid retrieval with entity resolution and abstention</p>
        </div>
        <div className="health">
          {health ? (
            <>
              <span className="dot ok" /> {health.storage_backend} · {health.embedding_model} ·{" "}
              {health.counts.documents} docs / {health.counts.chunks} chunks / {health.counts.entities} entities
            </>
          ) : (
            <>
              <span className="dot bad" /> API unreachable at {API_URL} {healthError ? `(${healthError})` : ""}
            </>
          )}
        </div>
      </header>
      <nav className="tabs">
        {(["search", "documents", "entities"] as Tab[]).map((t) => (
          <button key={t} className={tab === t ? "tab active" : "tab"} onClick={() => setTab(t)}>
            {t[0].toUpperCase() + t.slice(1)}
          </button>
        ))}
      </nav>
      <main>
        {tab === "search" && <SearchView />}
        {tab === "documents" && <DocumentsView onChange={refreshHealth} />}
        {tab === "entities" && <EntitiesView />}
      </main>
    </div>
  );
}
