import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  BookOpen,
  Bot,
  CheckSquare,
  Clapperboard,
  Database,
  Eye,
  FileText,
  Home,
  Loader2,
  Pause,
  Play,
  ScrollText,
  Search,
  Settings,
  Sparkles,
  Upload,
  Users,
  Zap,
} from "lucide-react";

// ------------------------------------------------------------------------------------
// API contract
// ------------------------------------------------------------------------------------
interface SceneFeatures {
  scene_title: string;
  duration_sec: number;
  dialogue_ratio: number;
  pacing_wpm: number;
  motion_intensity: number;
  sentiment: string;
  action_level: number;
  camera_motion: string;
  music_intensity: number;
  emotional_intensity: number;
  avg_shot_length_sec: number;
  visible_characters: number;
  summary: string;
}
interface RiskScan {
  level: "HIGH" | "MEDIUM" | "LOW";
  matched: number;
  total: number;
  factors: { label: string; matched: boolean }[];
  statement: string;
}
interface Segment {
  age_group: string;
  device: string;
  sample_size: number;
  drops: number;
  dropout_percentage: number;
  hazard_ratio: number;
}
interface CurveBucket {
  bucket: number;
  rel_start: number;
  rel_end: number;
  views: number;
  drops: number;
  drop_pct: number;
}
interface Pattern {
  pattern: string;
  sample_size: number;
  dropout_percentage: number;
  hazard_ratio: number;
}
interface ScatterPoint {
  scene_id: string;
  scene_title: string;
  duration_sec: number;
  dialogue_ratio: number;
  age_group: string;
  device: string;
  views: number;
  exit_rate: number;
}
interface SceneType {
  scene_type: string;
  views: number;
  drops: number;
  exit_rate: number;
}
interface ClickHouseEvidence {
  transport: string;
  mcp_server: string;
  fallback: boolean;
  fallback_reason?: string;
  latency_ms: number;
  query_latency_ms: { segments: number; curve: number; count: number };
  rows_scanned: number;
  sql: string;
  bound_sql: string;
  curve_sql: string;
  parameters: Record<string, number | string>;
  segments: Segment[];
  top_risk: Segment | null;
  curve: CurveBucket[];
  patterns: Pattern[];
  scatter: ScatterPoint[];
  scene_types: SceneType[];
  scene_count: number;
  match_note?: string;
  matched_scenes?: number;
  tables: string[];
}
interface CutRecommendation {
  start_sec: number;
  end_sec: number;
  action: string;
  rationale: string;
}
interface PrescriptionStructured {
  headline: string;
  risk_summary: string;
  cuts: CutRecommendation[];
  camera_directions: string[];
  predicted_drop_pct_after: number;
  confidence: number;
  report_text: string;
}
interface TraceStep {
  step: number;
  actor: string;
  tool: string;
  status: string;
  latency_ms: number;
  rows: number | null;
  detail: string;
}
interface AnalysisResponse {
  scene_id: string;
  source: string;
  mode: { mode: "live" | "partial" | "demo"; label: string; detail: string };
  risk: RiskScan;
  features: SceneFeatures;
  clickhouse: ClickHouseEvidence;
  prescription: string;
  prescription_structured: PrescriptionStructured;
  agent_trace: TraceStep[];
  total_latency_ms: number;
  model: string;
  cached?: boolean;
  upload?: { filename: string; bytes: number; mime: string };
}
interface HealthResponse {
  status: string;
  clickhouse: boolean;
  rows_scanned: number;
  gemini: boolean;
  gemini_model: string;
  mcp: { connected: boolean; server: string; tools: string[]; host: string };
}
interface AskResponse {
  question: string;
  answer: string;
  queries: { sql: string; columns: string[]; rows: unknown[][]; latency_ms: number }[];
  agent_trace: TraceStep[];
  total_latency_ms: number;
  model: string;
}
interface RecentAnalysis {
  title: string;
  risk: "High" | "Medium" | "Low";
  at: number;
  scene_id: string;
}

// ------------------------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------------------------
const tc = (seconds: number): string => {
  const s = Math.max(0, Math.round(seconds));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
};
const fmtRows = (n: number): string => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}K` : String(n));
const fmtMs = (ms: number): string => (ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms.toFixed(1)}ms`);
const clock = (d: Date) => `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
const ago = (ts: number) => {
  const m = Math.round((Date.now() - ts) / 60000);
  return m < 1 ? "just now" : m < 60 ? `${m}m ago` : `${Math.round(m / 60)}h ago`;
};
type Risk = "High" | "Medium" | "Low";
const riskOf = (hazard: number): Risk => (hazard >= 2 ? "High" : hazard >= 1.3 ? "Medium" : "Low");
const riskTone: Record<Risk, string> = {
  High: "bg-orange-500/15 text-orange-300 border-orange-500/30",
  Medium: "bg-amber-400/15 text-amber-200 border-amber-400/30",
  Low: "bg-lime-400/15 text-lime-300 border-lime-400/30",
};
const actionLabel: Record<string, string> = { trim: "Trim", tighten: "Tighten", insert_cutaway: "Insert cutaway", reframe: "Reframe", reorder: "Reorder" };

const NAV = [
  { id: "home", label: "Home", icon: Home },
  { id: "analyze", label: "Analyze Scene", icon: Clapperboard },
  { id: "patterns", label: "Discover Patterns", icon: Sparkles },
  { id: "library", label: "Episodes & Library", icon: BookOpen },
  { id: "audience", label: "Audience Insights", icon: Users },
  { id: "reports", label: "Reports", icon: FileText },
  { id: "logs", label: "Agent Logs", icon: ScrollText, dot: true },
  { id: "settings", label: "Settings", icon: Settings },
];

const AGENTS = [
  { key: "scene", name: "Scene Analyst", role: "Gemini analyzes the video", color: "bg-lime-400" },
  { key: "pattern", name: "Pattern Analyst", role: "ClickHouse searches history via MCP", color: "bg-emerald-400" },
  { key: "risk", name: "Risk Analyst", role: "Compares scene against patterns", color: "bg-sky-400" },
  { key: "strategist", name: "Strategist", role: "Explains and recommends", color: "bg-amber-400" },
];
const agentForStep = (t: TraceStep) => {
  if (t.tool.startsWith("gemini.extract")) return AGENTS[0];
  if (t.tool.startsWith("risk.")) return AGENTS[2];
  if (t.tool.startsWith("gemini.synth")) return AGENTS[3];
  return AGENTS[1];
};

// ------------------------------------------------------------------------------------
export default function App() {
  const [data, setData] = useState<AnalysisResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [selectedCut, setSelectedCut] = useState("cut_01");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [nav, setNav] = useState<string>(() => {
    const h = window.location.hash.replace(/^#\/?/, "");
    return NAV.some((n) => n.id === h) ? h : "home";
  });
  const PAGE_SECTIONS: Record<string, string[]> = {
    home: ["home", "kpis", "analyze", "patterns", "reports"],
    analyze: ["kpis", "analyze", "reports"],
    patterns: ["kpis", "patterns", "library"],
    library: ["library"],
    audience: ["kpis", "library"],
    reports: ["reports", "logs"],
    logs: ["logs"],
    settings: ["settings"],
  };
  const show = (section: string) => (PAGE_SECTIONS[nav] ?? PAGE_SECTIONS.home).includes(section);
  const [now, setNow] = useState(new Date());
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [sqlView, setSqlView] = useState<"parameterized" | "bound" | "curve">("parameterized");
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [askResult, setAskResult] = useState<AskResponse | null>(null);
  const [evidenceInline, setEvidenceInline] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [recent, setRecent] = useState<RecentAnalysis[]>(() => {
    try {
      return JSON.parse(localStorage.getItem("scenedna.recent") || "[]") as RecentAnalysis[];
    } catch {
      return [];
    }
  });
  const [liveLog, setLiveLog] = useState<{ at: number; agent: (typeof AGENTS)[number]; text: string }[]>([]);
  const fileInput = useRef<HTMLInputElement | null>(null);
  const dragRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 15000);
    return () => clearInterval(id);
  }, []);

  const pushRecent = useCallback((r: AnalysisResponse) => {
    const hazard = r.clickhouse.top_risk?.hazard_ratio ?? 0;
    const entry: RecentAnalysis = { title: r.features.scene_title, risk: riskOf(hazard), at: Date.now(), scene_id: r.scene_id };
    setRecent((prev) => {
      const next = [entry, ...prev.filter((p) => p.title !== entry.title)].slice(0, 6);
      try {
        localStorage.setItem("scenedna.recent", JSON.stringify(next));
      } catch {
        /* ignore */
      }
      return next;
    });
  }, []);

  const applyResult = useCallback(
    (r: AnalysisResponse) => {
      setData(r);
      setCurrentTime(0);
      setIsPlaying(false);
      pushRecent(r);
      const base = Date.now();
      setLiveLog(
        r.agent_trace
          .filter((t) => t.status !== "retry")
          .map((t, i) => ({ at: base - (r.agent_trace.length - i) * 4000, agent: agentForStep(t), text: t.detail.replace(/\s+/g, " ").slice(0, 120) })),
      );
    },
    [pushRecent],
  );

  const fetchPreset = useCallback(
    async (sceneId: string) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch("/api/analyze-preset", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ scene_id: sceneId }) });
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `Request failed (${res.status})`);
        applyResult((await res.json()) as AnalysisResponse);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Could not reach the SceneDNA agent.");
      } finally {
        setLoading(false);
      }
    },
    [applyResult],
  );

  const uploadVideo = useCallback(
    async (file: File) => {
      setLoading(true);
      setError(null);
      try {
        const form = new FormData();
        form.append("file", file);
        setVideoUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return URL.createObjectURL(file);
        });
        const res = await fetch("/api/analyze-video", { method: "POST", body: form });
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `Upload failed (${res.status})`);
        setSelectedCut("upload");
        applyResult((await res.json()) as AnalysisResponse);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Upload failed.");
      } finally {
        setLoading(false);
      }
    },
    [applyResult],
  );

  const ask = useCallback(async () => {
    const q = question.trim();
    if (!q) return;
    setAsking(true);
    setAskResult(null);
    try {
      const res = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q }) });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `Ask failed (${res.status})`);
      const r = (await res.json()) as AskResponse;
      setAskResult(r);
      setLiveLog((prev) => [...r.agent_trace.filter((t) => t.status === "ok").map((t) => ({ at: Date.now(), agent: AGENTS[2], text: t.detail.slice(0, 120) })), ...prev].slice(0, 12));
    } catch (e) {
      setAskResult({ question: q, answer: e instanceof Error ? e.message : "Ask failed.", queries: [], agent_trace: [], total_latency_ms: 0, model: "" });
    } finally {
      setAsking(false);
    }
  }, [question]);

  useEffect(() => {
    fetchPreset("cut_01");
    fetch("/api/health")
      .then((r) => r.json())
      .then((h: HealthResponse) => setHealth(h))
      .catch(() => undefined);
  }, [fetchPreset]);

  const duration = data?.features.duration_sec ?? 0;
  useEffect(() => {
    if (videoUrl) return;
    if (!isPlaying || !duration) return;
    const id = setInterval(() => setCurrentTime((t) => (t + 1 >= duration ? 0 : t + 1)), 1000);
    return () => clearInterval(id);
  }, [isPlaying, duration, videoUrl]);
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !videoUrl) return;
    if (isPlaying) v.play().catch(() => setIsPlaying(false));
    else v.pause();
  }, [isPlaying, videoUrl]);

  // -- derived --------------------------------------------------------------------
  const ch = data?.clickhouse;
  const top = ch?.top_risk ?? null;
  const globalRate = useMemo(() => {
    const segs = ch?.segments ?? [];
    const v = segs.reduce((a, s) => a + s.sample_size, 0);
    const d = segs.reduce((a, s) => a + s.drops, 0);
    return v ? (d / v) * 100 : 8;
  }, [ch]);
  const risk: Risk = data ? ((data.risk.level.charAt(0) + data.risk.level.slice(1).toLowerCase()) as Risk) : "Low";
  const factors = data ? data.risk.factors.map((f) => ({ label: f.label, hit: f.matched })) : [];
  const matched = data?.risk.matched ?? 0;
  const positiveFactor: Record<string, string> = {
    "Dialogue ratio above 75%": "Dialogue ratio below the risk threshold",
    "Duration above 4 minutes": "Runtime under 4 minutes",
    "Low camera motion": "Camera motion above the risk threshold",
    "Low music intensity": "Music intensity above the risk threshold",
  };
  const recCount = (data?.prescription_structured.cuts.length ?? 0) + (data?.prescription_structured.camera_directions.length ?? 0);
  const topPattern = ch?.patterns?.[0];
  const insight = data && top
    ? matched >= 3
      ? `This scene matches ${matched} of ${factors.length} patterns historically associated with elevated abandonment among ${top.age_group} ${top.device} viewers.`
      : topPattern
        ? `Historical insight: ${topPattern.pattern.toLowerCase()} is associated with ${topPattern.hazard_ratio.toFixed(1)}x the baseline exit rate among ${top.age_group} ${top.device} viewers. This scene avoids that pattern.`
        : ""
    : "";
  const agentStats: Record<string, string> = data
    ? {
        scene: `${Object.keys(data.features).length} features extracted`,
        pattern: `${fmtRows(ch?.rows_scanned ?? 0)} events · ${ch?.matched_scenes ?? 0} similar scenes compared`,
        risk: `${matched}/${factors.length} risk factors matched`,
        strategist: `${recCount} recommendations generated`,
      }
    : {};
  const cuts = data?.prescription_structured.cuts ?? [];
  const scrollTo = (id: string) => {
    setNav(id);
    window.location.hash = `/${id}`;
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  useEffect(() => {
    const onHash = () => {
      const h = window.location.hash.replace(/^#\/?/, "");
      if (NAV.some((n) => n.id === h)) setNav(h);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  // ----------------------------------------------------------------------------------
  return (
    <div className="flex min-h-screen bg-[#070907] text-slate-200">
      {/* Sidebar */}
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r border-white/5 bg-[#0a0d0a] lg:flex">
        <div className="flex items-center gap-3 px-5 py-5">
          <div className="grid h-9 w-9 place-items-center rounded-lg border border-lime-400/40 bg-lime-400/10">
            <Clapperboard className="h-5 w-5 text-lime-300" />
          </div>
          <div>
            <div className="text-xl font-bold leading-none tracking-tight text-white">
              Scene<span className="text-lime-300">DNA</span>
            </div>
            <div className="mt-1 text-[10px] uppercase tracking-widest text-slate-500">AI for better stories</div>
          </div>
        </div>
        <nav className="mt-2 flex flex-col gap-1 px-3">
          {NAV.map((n) => (
            <button key={n.id} onClick={() => scrollTo(n.id)} className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition ${nav === n.id ? "bg-lime-400/10 text-lime-200 ring-1 ring-lime-400/30" : "text-slate-400 hover:bg-white/5 hover:text-white"}`}>
              <n.icon className="h-4 w-4" />
              {n.label}
              {n.dot && <span className="ml-auto h-1.5 w-1.5 rounded-full bg-lime-400" />}
            </button>
          ))}
        </nav>
        <div className="mt-auto space-y-4 px-5 pb-5">
          <div className="rounded-xl bg-gradient-to-br from-[#121812] to-[#0a0d0a] p-4 text-[11px] uppercase leading-relaxed tracking-[0.25em] text-slate-400">
            Data sees what stories feel.
          </div>
          <div className="flex items-center gap-3 rounded-xl border border-white/5 p-3">
            <div className="grid h-9 w-9 place-items-center rounded-full bg-sky-500/20 font-semibold text-sky-300">R</div>
            <div className="text-sm">
              <div className="text-white">Roshni K</div>
              <div className="text-xs text-slate-500">Studio workspace</div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main */}
      <div className="min-w-0 flex-1">
        {/* Top bar */}
        <header className="sticky top-0 z-20 flex items-center gap-3 border-b border-white/5 bg-[#070907]/90 px-6 py-3 backdrop-blur">
          <div className="flex flex-1 items-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2">
            <Search className="h-4 w-4 text-slate-500" />
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && ask()}
              placeholder="Ask anything... (e.g. Which scenes lose young mobile viewers?)"
              className="flex-1 bg-transparent text-sm text-white placeholder:text-slate-500 focus:outline-none"
            />
            <button onClick={ask} disabled={asking || !question.trim()} className="flex items-center gap-1 rounded-md bg-lime-400 px-2.5 py-1 text-xs font-semibold text-black disabled:opacity-40">
              {asking ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ArrowRight className="h-3.5 w-3.5" />} Ask
            </button>
          </div>
          <div className="hidden items-center gap-3 font-mono text-[10px] uppercase tracking-wider text-slate-400 xl:flex">
            <span className={health?.gemini ? "text-lime-300" : ""}>Gemini video {health?.gemini ? "✓" : "…"}</span>
            <span className={health?.mcp.connected ? "text-lime-300" : ""}>ClickHouse MCP {health?.mcp.connected ? "✓" : "…"}</span>
            <span className={health?.gemini ? "text-lime-300" : ""}>{(data?.model || health?.gemini_model || "gemini").replace("gemini-", "Gemini ")} ✓</span>
          </div>
          {data && (
            <div title={data.mode.detail} className={`hidden items-center gap-2 rounded-xl border px-3 py-2 font-mono text-[11px] uppercase tracking-wider md:flex ${data.mode.mode === "live" ? "border-lime-400/40 bg-lime-400/10 text-lime-300" : data.mode.mode === "partial" ? "border-amber-400/40 bg-amber-400/10 text-amber-200" : "border-orange-500/40 bg-orange-500/10 text-orange-300"}`}>
              <span className={`h-2 w-2 rounded-full ${data.mode.mode === "live" ? "bg-lime-400" : "bg-amber-400 animate-pulse"}`} />
              {data.mode.label}
            </div>
          )}
          <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2 text-sm">
            <span className={`h-2 w-2 rounded-full ${health?.clickhouse && health?.gemini ? "bg-lime-400" : "bg-amber-400 animate-pulse"}`} />
            Studio Mode
            <span className="font-mono text-xs text-slate-500">{clock(now)}</span>
          </div>
        </header>

        <main className="grid gap-5 px-6 py-5 2xl:grid-cols-[minmax(0,1fr)_320px]">
          <div className="min-w-0 space-y-5">
            {nav !== "home" && (
              <div className="flex items-center gap-2 text-xs text-slate-500">
                <button onClick={() => scrollTo("home")} className="hover:text-white">Home</button>
                <span>/</span>
                <span className="text-white">{NAV.find((n) => n.id === nav)?.label}</span>
              </div>
            )}
            {data && data.mode.mode !== "live" && (
              <div className="flex items-center gap-3 rounded-xl border border-amber-400/30 bg-amber-400/10 px-4 py-2.5 text-xs text-amber-100">
                <AlertTriangle className="h-4 w-4 shrink-0" /> {data.mode.label}: {data.mode.detail}. Live results return automatically once the service recovers.
                <button onClick={() => fetchPreset(selectedCut === "upload" ? "cut_01" : selectedCut)} className="ml-auto rounded-md border border-amber-300/30 px-2 py-1">Retry live</button>
              </div>
            )}
            {error && (
              <div className="flex items-center gap-3 rounded-xl border border-orange-500/30 bg-orange-500/10 px-4 py-3 text-sm text-orange-200">
                <AlertTriangle className="h-4 w-4" /> {error}
                <button onClick={() => fetchPreset(selectedCut === "upload" ? "cut_01" : selectedCut)} className="ml-auto rounded-md border border-orange-400/30 px-2 py-1 text-xs">Retry</button>
              </div>
            )}

            {askResult && (
              <div className="rounded-2xl border border-lime-400/20 bg-lime-400/5 p-4">
                <div className="flex items-center gap-2 text-xs uppercase tracking-widest text-lime-300">
                  <Bot className="h-4 w-4" /> Agent answer · {askResult.queries.length} ClickHouse queries over MCP · {fmtMs(askResult.total_latency_ms)}
                  <button onClick={() => setAskResult(null)} className="ml-auto text-slate-500 hover:text-white">close</button>
                </div>
                <p className="mt-2 text-sm leading-relaxed text-slate-200">{askResult.answer}</p>
                {askResult.queries.length > 0 && (
                  <details className="mt-2">
                    <summary className="cursor-pointer text-xs text-slate-500">Show SQL the agent ran</summary>
                    {askResult.queries.map((q, i) => (
                      <pre key={i} className="mt-2 overflow-auto rounded-lg bg-black/40 p-3 font-mono text-[11px] text-emerald-200">{q.sql}</pre>
                    ))}
                  </details>
                )}
              </div>
            )}

            {/* Hero */}
            <section id="sec-home" hidden={!show("home")} className="relative overflow-hidden rounded-2xl border border-white/5 bg-[#0c100c]">
              <div className="pointer-events-none absolute inset-0 [background:radial-gradient(ellipse_at_80%_40%,rgba(251,146,60,0.22),transparent_45%),radial-gradient(ellipse_at_95%_90%,rgba(163,230,53,0.12),transparent_40%)]" />
              <div className="pointer-events-none absolute right-0 top-0 h-full w-1/2 bg-[linear-gradient(120deg,transparent_0%,rgba(255,255,255,0.03)_50%,transparent_100%)]" />
              <div className="relative grid gap-6 p-8 md:grid-cols-[1.4fr_1fr]">
                <div>
                  <p className="text-[10px] uppercase tracking-[0.35em] text-slate-500">Turn scene data into bigger stories</p>
                  <h1 className="mt-3 text-3xl font-bold leading-tight text-white md:text-4xl">
                    Find the Creative Patterns
                    <br />
                    Behind <span className="text-lime-300">Audience Drop-Off</span>
                  </h1>
                  <p className="mt-3 max-w-lg text-sm text-slate-400">Gemini watches scenes. ClickHouse finds the patterns. SceneDNA flags risky moments before release.</p>
                  <div className="mt-5 flex gap-3">
                    <button onClick={() => scrollTo("analyze")} className="flex items-center gap-2 rounded-lg bg-lime-400 px-4 py-2.5 text-sm font-semibold text-black hover:bg-lime-300">
                      Analyze a Scene <ArrowRight className="h-4 w-4" />
                    </button>
                    <button onClick={() => scrollTo("patterns")} className="rounded-lg border border-white/15 px-4 py-2.5 text-sm text-white hover:bg-white/5">Discover Patterns</button>
                  </div>
                </div>
                <div className="hidden items-end justify-end md:flex">
                  <div className="text-right text-[10px] uppercase leading-relaxed tracking-[0.3em] text-slate-400">
                    "Same footage.
                    <br />
                    Deeper insights."
                  </div>
                </div>
              </div>
            </section>

            {/* KPIs */}
            <section id="sec-kpis" hidden={!show("kpis")} className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <Kpi icon={<Eye className="h-5 w-5" />} value={fmtRows(health?.rows_scanned ?? ch?.rows_scanned ?? 0)} label="Viewer Events" spark={[3, 4, 4, 5, 6, 7, 8, 9]} tone="lime" />
              <Kpi icon={<Database className="h-5 w-5" />} value={String(ch?.scene_count ?? 0)} label="Scenes Analyzed" spark={[2, 3, 3, 4, 5, 5, 6, 6]} tone="sky" />
              <Kpi icon={<Users className="h-5 w-5" />} value={String(ch?.segments.length ?? 0)} label="Audience Segments" spark={[4, 4, 5, 5, 5, 6, 6, 7]} tone="violet" />
              <Kpi icon={<Zap className="h-5 w-5" />} value={ch ? fmtMs(ch.latency_ms) : "…"} label="Avg Query Latency" spark={[6, 4, 5, 3, 4, 3, 2, 2]} tone="lime" />
            </section>

            {/* Scene Analysis */}
            <section id="sec-analyze" hidden={!show("analyze")} className="rounded-2xl border border-white/5 bg-[#0c100c] p-5">
              <div className="mb-4 flex flex-wrap items-center gap-2">
                <div>
                  <h2 className="text-lg font-semibold text-white">Scene Analysis</h2>
                  <p className="text-xs text-slate-500">Upload or select a scene and let Gemini extract creative DNA</p>
                </div>
                <span className="ml-auto text-xs text-slate-500">
                  Powered by <span className="text-lime-300">{data?.model || health?.gemini_model || "Gemini"}</span> + <span className="text-amber-300">ClickHouse</span>
                </span>
              </div>
              <div className="grid gap-4 lg:grid-cols-[1fr_1.4fr_1fr]">
                {/* Upload */}
                <div
                  ref={dragRef}
                  onDragOver={(e) => {
                    e.preventDefault();
                    dragRef.current?.classList.add("border-lime-400");
                  }}
                  onDragLeave={() => dragRef.current?.classList.remove("border-lime-400")}
                  onDrop={(e) => {
                    e.preventDefault();
                    dragRef.current?.classList.remove("border-lime-400");
                    const f = e.dataTransfer.files?.[0];
                    if (f) uploadVideo(f);
                  }}
                  onClick={() => fileInput.current?.click()}
                  className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-white/15 bg-white/[0.02] p-6 text-center transition hover:border-lime-400/60"
                >
                  <div className="grid h-14 w-14 place-items-center rounded-full bg-sky-500/15 text-sky-300">
                    <Upload className="h-6 w-6" />
                  </div>
                  <p className="mt-4 text-sm text-white">Drop a video file here or click to upload</p>
                  <p className="mt-2 text-xs text-slate-500">Supports MP4, MOV, WEBM (max 30MB)</p>
                  <input ref={fileInput} type="file" accept="video/mp4,video/quicktime,video/webm,video/x-matroska" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) uploadVideo(f); e.target.value = ""; }} />
                  <div className="mt-5 flex gap-2" onClick={(e) => e.stopPropagation()}>
                    {["cut_01", "cut_02"].map((id) => (
                      <button key={id} disabled={loading} onClick={() => { setSelectedCut(id); setVideoUrl(null); fetchPreset(id); }} className={`rounded-md border px-2.5 py-1 font-mono text-[11px] ${selectedCut === id ? "border-lime-400/50 bg-lime-400/10 text-lime-200" : "border-white/10 text-slate-400 hover:text-white"}`}>
                        {id === "cut_01" ? "Preset · Dialogue lock" : "Preset · Action cut"}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Monitor */}
                <div className="overflow-hidden rounded-xl border border-white/10 bg-black">
                  <div className="relative aspect-video bg-gradient-to-br from-[#1a1f2b] via-[#0b0e14] to-black">
                    {videoUrl && (
                      <video
                        ref={videoRef}
                        src={videoUrl}
                        playsInline
                        className="absolute inset-0 h-full w-full object-contain"
                        onTimeUpdate={(e) => setCurrentTime(Math.floor(e.currentTarget.currentTime))}
                        onEnded={() => setIsPlaying(false)}
                        onClick={() => setIsPlaying((p) => !p)}
                      />
                    )}
                    <div className="pointer-events-none absolute inset-0 [background:radial-gradient(circle_at_70%_30%,rgba(251,191,36,0.18),transparent_40%)]" />
                    <div className={`pointer-events-none absolute inset-0 transition-opacity ${isPlaying ? "opacity-100 animate-drift" : "opacity-0"} [background:radial-gradient(circle_at_30%_60%,rgba(255,255,255,0.06),transparent_40%)]`} />
                    <div className={`absolute inset-0 flex flex-col items-center justify-center gap-2 px-4 text-center transition-opacity ${videoUrl && isPlaying && !loading ? "pointer-events-none opacity-0" : ""}`}>
                      {loading ? (
                        <>
                          <Loader2 className="h-7 w-7 animate-spin text-lime-300" />
                          <p className="text-xs uppercase tracking-widest text-lime-200">Agents working</p>
                        </>
                      ) : (
                        <>
                          <button onClick={() => setIsPlaying((p) => !p)} className="grid h-14 w-14 place-items-center rounded-full bg-white/90 text-black shadow-xl hover:scale-105">
                            {isPlaying ? <Pause className="h-6 w-6" /> : <Play className="ml-1 h-6 w-6" />}
                          </button>
                          <p className={`mt-2 text-sm text-white ${videoUrl ? "rounded bg-black/60 px-2 py-0.5" : ""}`}>{data?.features.scene_title}</p>
                        </>
                      )}
                    </div>
                    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent px-3 pb-2 pt-6">
                      <div className="relative h-1 w-full rounded bg-white/20">
                        {ch?.curve.map((b) => {
                          const r = b.drop_pct / globalRate;
                          return <div key={b.bucket} className={`absolute top-0 h-full ${r >= 2 ? "bg-orange-500" : r >= 1.3 ? "bg-amber-400" : "bg-lime-400/70"}`} style={{ left: `${b.rel_start * 100}%`, width: "5%" }} />;
                        })}
                        {cuts.map((c, i) => (
                          <div key={i} className="absolute -top-1 h-3 border-x border-white bg-white/30" style={{ left: `${(c.start_sec / duration) * 100}%`, width: `${((c.end_sec - c.start_sec) / duration) * 100}%` }} />
                        ))}
                        <div className="absolute -top-1 h-3 w-3 -translate-x-1/2 rounded-full bg-white" style={{ left: `${duration ? (currentTime / duration) * 100 : 0}%` }} />
                      </div>
                      <div className="mt-1 flex items-center gap-3 font-mono text-[11px] text-slate-300">
                        <button onClick={() => setIsPlaying((p) => !p)}>{isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}</button>
                        {tc(currentTime)} / {tc(duration)}
                        <span className="ml-auto text-slate-500">{data?.source === "video" ? "uploaded" : "preset"}</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Features */}
                <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
                  <p className="mb-3 text-xs font-semibold text-white">Extracted Scene Features</p>
                  {data ? (
                    <div className="space-y-2.5 text-xs">
                      <Feature label="Dialogue Ratio" pct={data.features.dialogue_ratio * 100} value={`${Math.round(data.features.dialogue_ratio * 100)}%`} />
                      <Feature label="Action Level" pct={data.features.action_level * 100} value={`${Math.round(data.features.action_level * 100)}%`} />
                      <Feature label="Camera Motion" pct={data.features.motion_intensity * 100} value={data.features.camera_motion.charAt(0).toUpperCase() + data.features.camera_motion.slice(1)} />
                      <Feature label="Music Intensity" pct={data.features.music_intensity * 100} value={data.features.music_intensity < 0.3 ? "Low" : data.features.music_intensity < 0.6 ? "Medium" : "High"} />
                      <Feature label="Emotional Intensity" pct={data.features.emotional_intensity * 100} value={data.features.emotional_intensity.toFixed(1)} />
                      <Feature label="Pacing" pct={Math.min(100, (data.features.pacing_wpm / 260) * 100)} value={`${data.features.pacing_wpm} wpm`} />
                      <Row label="Scene Duration" value={`${Math.floor(duration / 60)}m ${duration % 60}s`} />
                      <Row label="Avg. Shot Length" value={`${data.features.avg_shot_length_sec.toFixed(1)}s`} />
                      <Row label="Characters Visible" value={String(data.features.visible_characters)} />
                      <Row label="Scene Type" value={data.features.sentiment.replace(/_/g, " ")} />
                      {data.features.summary && <p className="pt-1 text-[11px] leading-relaxed text-slate-400">{data.features.summary}</p>}
                    </div>
                  ) : (
                    <div className="space-y-2">{Array.from({ length: 6 }).map((_, i) => <div key={i} className="h-3 animate-pulse rounded bg-white/5" />)}</div>
                  )}
                </div>
              </div>
            </section>

            {/* Pattern discovery */}
            <section id="sec-patterns" hidden={!show("patterns")} className="rounded-2xl border border-white/5 bg-[#0c100c] p-5">
              <div className="mb-4 flex flex-wrap items-center gap-3">
                <div>
                  <h2 className="text-lg font-semibold text-white">Historical Pattern Discovery</h2>
                  <p className="text-xs text-slate-500">Find what creative patterns are associated with audience drop-off</p>
                </div>
                <div className="ml-auto flex flex-wrap gap-2 text-[11px]">
                  <Chip label="Audience Segment" value={top ? `${top.age_group}` : "—"} />
                  <Chip label="Device Type" value={top ? top.device : "—"} />
                  <Chip label="Source" value={ch?.fallback ? "offline" : "ClickHouse MCP"} />
                </div>
              </div>
              <div className="grid gap-4 xl:grid-cols-[1.1fr_1.3fr_1fr]">
                <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
                  <p className="mb-3 text-xs font-semibold text-white">Top Risk Patterns</p>
                  <ol className="space-y-2">
                    {(ch?.patterns ?? []).slice(0, 5).map((p, i) => (
                      <li key={p.pattern} className="flex items-center gap-3 rounded-lg bg-white/[0.03] px-3 py-2 text-xs">
                        <span className={`grid h-5 w-5 place-items-center rounded-full font-mono text-[10px] ${i < 2 ? "bg-orange-500 text-black" : "bg-amber-400 text-black"}`}>{i + 1}</span>
                        <span className="flex-1 text-slate-200">{p.pattern}</span>
                        <span className={`rounded-md border px-2 py-0.5 font-mono ${riskTone[riskOf(p.hazard_ratio)]}`}>{p.hazard_ratio.toFixed(1)}x exits</span>
                      </li>
                    ))}
                  </ol>
                </div>
                <Scatter points={ch?.scatter ?? []} highlight={top} />
                <Donut types={ch?.scene_types ?? []} />
              </div>
            </section>

            {/* Risk assessment */}
            <section id="sec-reports" hidden={!show("reports")} className="rounded-2xl border border-white/5 bg-[#0c100c] p-5">
              <div className="mb-4 flex flex-wrap items-center gap-3">
                <h2 className="text-lg font-semibold text-white">New Scene Risk Assessment</h2>
                <span className="text-xs text-slate-500">Compare your scene against discovered patterns</span>
                <button onClick={() => setEvidenceInline((v) => !v)} className="ml-auto flex items-center gap-1 rounded-md border border-amber-400/40 bg-amber-400/10 px-3 py-1.5 text-xs text-amber-200 hover:bg-amber-400/20">
                  <Database className="h-3.5 w-3.5" /> {evidenceInline ? "Hide" : "View"} ClickHouse evidence
                </button>
              </div>
              {insight && (
                <div className={`mb-4 rounded-xl border px-4 py-3 text-sm ${matched >= 3 ? "border-orange-500/30 bg-orange-500/10 text-orange-100" : "border-lime-400/30 bg-lime-400/10 text-lime-100"}`}>
                  {insight}
                </div>
              )}
              {evidenceInline && ch && (
                <div className="mb-4 grid gap-3 rounded-xl border border-amber-400/20 bg-black/30 p-4 lg:grid-cols-[1.4fr_1fr]">
                  <pre className="max-h-56 overflow-auto rounded-lg bg-black/50 p-3 font-mono text-[11px] leading-relaxed text-emerald-200">{ch.bound_sql}</pre>
                  <div className="space-y-2 text-xs">
                    <Row label="Executed via" value={`${ch.transport} · ${ch.mcp_server}`} />
                    <Row label="Events scanned" value={`${ch.rows_scanned.toLocaleString()} rows in ${fmtMs(ch.latency_ms)}`} />
                    <Row label="Scenes queried" value={`${ch.matched_scenes ?? 0} similar of ${ch.scene_count} in library`} />
                    <Row label="Top segment" value={top ? `${top.age_group} ${top.device} · ${top.dropout_percentage.toFixed(2)}% · ${top.hazard_ratio.toFixed(2)}x` : "—"} />
                    <Row label="Risk pattern" value={topPattern ? `${topPattern.pattern} · ${topPattern.hazard_ratio.toFixed(1)}x` : "—"} />
                    <button onClick={() => scrollTo("logs")} className="mt-1 text-[11px] text-amber-200 hover:text-white">Full agent trace and every query →</button>
                  </div>
                </div>
              )}
              <div className="grid gap-4 lg:grid-cols-[1.3fr_1fr_1.3fr]">
                <div className={`flex items-center gap-4 rounded-xl border p-4 ${risk === "High" ? "border-orange-500/40 bg-orange-500/10" : risk === "Medium" ? "border-amber-400/40 bg-amber-400/10" : "border-lime-400/40 bg-lime-400/10"}`}>
                  <div className={`grid h-12 w-12 place-items-center rounded-xl ${risk === "High" ? "bg-orange-500/20 text-orange-300" : risk === "Medium" ? "bg-amber-400/20 text-amber-200" : "bg-lime-400/20 text-lime-300"}`}>
                    {risk === "Low" ? <CheckSquare className="h-6 w-6" /> : <AlertTriangle className="h-6 w-6" />}
                  </div>
                  <div className="flex-1">
                    <div className={`text-xl font-bold uppercase ${risk === "High" ? "text-orange-300" : risk === "Medium" ? "text-amber-200" : "text-lime-300"}`}>{risk} risk</div>
                    <div className="text-xs text-slate-400">{matched} of {factors.length} historical risk patterns matched</div>
                    <div className="mt-1 text-[11px] text-slate-500">Based on {ch?.matched_scenes ?? 0} similar scenes and {fmtRows(ch?.rows_scanned ?? 0)} viewer events in the SceneDNA library</div>
                    {ch?.match_note && <div className="mt-1 text-[11px] text-slate-500">{ch.match_note}</div>}
                  </div>
                  <div className="text-right">
                    <div className="text-[10px] uppercase tracking-widest text-slate-500">Estimated historical exit risk</div>
                    <div className={`font-mono text-3xl ${risk === "High" ? "text-orange-300" : "text-lime-300"}`}>
                      {top ? top.dropout_percentage.toFixed(0) : "—"}% <span className="text-base text-slate-400">± {top ? Math.max(2, Math.round(top.dropout_percentage * 0.15)) : 0}%</span>
                    </div>
                    <div className="text-[10px] text-slate-500">{top ? `${top.age_group} ${top.device} cohort, ${top.hazard_ratio.toFixed(2)}x baseline` : ""}</div>
                  </div>
                </div>
                <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
                  <p className="mb-2 text-xs font-semibold text-white">Why this scene is {risk.toLowerCase()} risk</p>
                  <ul className="space-y-1.5 text-xs">
                    {factors.map((f) => (
                      <li key={f.label} className="flex items-center gap-2">
                        <span className={`grid h-4 w-4 place-items-center rounded text-[10px] ${f.hit ? "bg-orange-500 text-black" : "bg-lime-400 text-black"}`}>{f.hit ? "!" : "✓"}</span>
                        <span className={f.hit ? "text-orange-200" : "text-slate-200"}>{f.hit ? `${f.label} (matches a historical risk pattern)` : positiveFactor[f.label] ?? f.label}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
                  <p className="mb-2 text-xs font-semibold text-white">Recommendations</p>
                  {data ? (
                    <ul className="space-y-1.5 text-xs">
                      {cuts.map((c, i) => (
                        <li key={i} className="flex gap-2">
                          <span className="mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded bg-lime-400 text-[10px] text-black">✓</span>
                          <span className="text-slate-200">
                            {actionLabel[c.action] ?? c.action} {tc(c.start_sec)} to {tc(c.end_sec)}: {c.rationale}
                          </span>
                        </li>
                      ))}
                      {data.prescription_structured.camera_directions.slice(0, 2).map((d, i) => (
                        <li key={`c${i}`} className="flex gap-2">
                          <span className="mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded bg-lime-400 text-[10px] text-black">✓</span>
                          <span className="text-slate-200">{d}</span>
                        </li>
                      ))}
                      {cuts.length === 0 && data.prescription_structured.camera_directions.length === 0 && <li className="text-slate-400">Lock the cut. No changes required.</li>}
                    </ul>
                  ) : null}
                  {data && (
                    <details className="mt-3">
                      <summary className="cursor-pointer text-[11px] text-slate-500 hover:text-slate-300">Read the full studio memo</summary>
                      <p className="mt-2 whitespace-pre-line text-xs leading-relaxed text-slate-300">{data.prescription}</p>
                    </details>
                  )}
                </div>
              </div>
            </section>

            {/* Library + audience */}
            <section id="sec-library" hidden={!show("library")} className="grid gap-5 lg:grid-cols-2">
              <div className="rounded-2xl border border-white/5 bg-[#0c100c] p-5">
                <h2 className="text-sm font-semibold text-white">Episodes & Library</h2>
                <p className="mb-3 text-xs text-slate-500">Scenes in the DNA library and their exit rate for the risk cohort</p>
                <div className="space-y-1.5">
                  {uniqueScenes(ch?.scatter ?? [], top).map((s) => (
                    <div key={s.scene_id} className="flex items-center gap-3 rounded-lg bg-white/[0.03] px-3 py-2 text-xs">
                      <span className="font-mono text-slate-500">{s.scene_id}</span>
                      <span className="flex-1 text-slate-200">{s.scene_title}</span>
                      <span className="text-slate-500">{tc(s.duration_sec)}</span>
                      <span className={`rounded-md border px-2 py-0.5 font-mono ${riskTone[riskOf(s.exit_rate / globalRate)]}`}>{s.exit_rate.toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
              <div id="sec-audience" className="rounded-2xl border border-white/5 bg-[#0c100c] p-5">
                <h2 className="text-sm font-semibold text-white">Audience Insights</h2>
                <p className="mb-3 text-xs text-slate-500">Hazard ratio by segment and device for scenes sharing this profile</p>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="text-[10px] uppercase tracking-widest text-slate-500">
                      <tr><th className="py-1.5">Segment</th><th>Device</th><th className="text-right">Sample</th><th className="text-right">Drop rate</th><th className="text-right">Hazard</th></tr>
                    </thead>
                    <tbody>
                      {(ch?.segments ?? []).slice(0, 8).map((s) => (
                        <tr key={`${s.age_group}-${s.device}`} className={`border-t border-white/5 ${s.hazard_ratio >= 2 ? "bg-orange-500/10 text-orange-100" : "text-slate-300"}`}>
                          <td className="py-1.5 font-mono">{s.age_group}</td>
                          <td className="capitalize">{s.device}</td>
                          <td className="text-right font-mono text-slate-500">{s.sample_size.toLocaleString()}</td>
                          <td className="text-right font-mono">{s.dropout_percentage.toFixed(2)}%</td>
                          <td className={`text-right font-mono ${s.hazard_ratio >= 2 ? "text-orange-300" : s.hazard_ratio >= 1.3 ? "text-amber-300" : "text-lime-300"}`}>{s.hazard_ratio.toFixed(2)}x</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>

            {/* Agent logs / evidence */}
            <section id="sec-logs" hidden={!show("logs")} className="rounded-2xl border border-white/5 bg-[#0c100c] p-5">
              <div className="mb-4 flex flex-wrap items-center gap-3">
                <Database className="h-4 w-4 text-amber-300" />
                <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-white">Agent Logs · ClickHouse evidence</h2>
                {ch && (
                  <span className="ml-auto flex flex-wrap gap-2 font-mono text-[11px] text-slate-400">
                    <span className="rounded-md border border-white/10 px-2 py-0.5">{fmtMs(ch.latency_ms)} hazard query</span>
                    <span className="rounded-md border border-white/10 px-2 py-0.5">{fmtRows(ch.rows_scanned)} rows</span>
                    <span className={`rounded-md border px-2 py-0.5 ${ch.fallback ? "border-amber-400/40 text-amber-300" : "border-lime-400/30 text-lime-300"}`}>{ch.transport} · {ch.mcp_server}</span>
                  </span>
                )}
              </div>
              {data && ch && (
                <div className="grid gap-5 lg:grid-cols-2">
                  <div>
                    <p className="mb-2 text-[10px] uppercase tracking-widest text-slate-500">Tool trace · {data.agent_trace.length} steps · {fmtMs(data.total_latency_ms)} total{data.cached ? " · cached" : ""}</p>
                    <ol className="max-h-80 space-y-1.5 overflow-auto pr-1">
                      {data.agent_trace.map((t) => (
                        <li key={`${t.step}-${t.tool}-${t.latency_ms}`} className="flex items-start gap-2 rounded-lg bg-white/[0.03] px-3 py-2 text-[11px]">
                          <span className={`rounded px-1.5 py-0.5 font-mono uppercase ${t.actor === "gemini" ? "bg-sky-500/15 text-sky-200" : "bg-amber-500/15 text-amber-200"}`}>{t.actor}</span>
                          <div className="min-w-0 flex-1">
                            <div className="flex flex-wrap gap-2">
                              <span className="font-mono text-slate-200">{t.tool}</span>
                              <span className={`rounded px-1.5 uppercase ${t.status === "ok" ? "bg-lime-400/15 text-lime-300" : t.status === "retry" || t.status === "skipped" ? "bg-amber-400/15 text-amber-200" : "bg-orange-500/15 text-orange-300"}`}>{t.status}</span>
                              {t.latency_ms > 0 && <span className="font-mono text-slate-500">{fmtMs(t.latency_ms)}</span>}
                              {t.rows !== null && <span className="font-mono text-slate-500">{t.rows} rows</span>}
                            </div>
                            <p className="truncate text-slate-400" title={t.detail}>{t.detail}</p>
                          </div>
                        </li>
                      ))}
                    </ol>
                  </div>
                  <div>
                    <div className="mb-2 flex items-center gap-2 text-[10px] uppercase tracking-widest text-slate-500">
                      SQL sent through mcp-clickhouse run_query
                      <div className="ml-auto flex gap-1">
                        {(["parameterized", "bound", "curve"] as const).map((v) => (
                          <button key={v} onClick={() => setSqlView(v)} className={`rounded px-2 py-0.5 ${sqlView === v ? "bg-white/10 text-white" : "hover:bg-white/5"}`}>{v}</button>
                        ))}
                      </div>
                    </div>
                    <pre className="max-h-80 overflow-auto rounded-lg bg-black/50 p-3 font-mono text-[11px] leading-relaxed text-emerald-200">{sqlView === "parameterized" ? ch.sql : sqlView === "bound" ? ch.bound_sql : ch.curve_sql}</pre>
                    <p className="mt-1 font-mono text-[10px] text-slate-500">parameters: {Object.entries(ch.parameters).map(([k, v]) => `${k}=${v}`).join("  ")}</p>
                  </div>
                </div>
              )}
            </section>

            <section id="sec-settings" hidden={!show("settings")} className="rounded-2xl border border-white/5 bg-[#0c100c] p-5 text-xs text-slate-400">
              <h2 className="mb-2 text-sm font-semibold text-white">Settings</h2>
              <div className="grid gap-2 sm:grid-cols-2">
                <Row label="Gemini model" value={data?.model || health?.gemini_model || "—"} />
                <Row label="ClickHouse host" value={health?.mcp.host || "—"} />
                <Row label="MCP server" value={health?.mcp.server || "—"} />
                <Row label="MCP tools" value={health?.mcp.tools.join(", ") || "—"} />
              </div>
            </section>
          </div>

          {/* Right rail */}
          <aside className="space-y-5">
            <div className="rounded-2xl border border-white/5 bg-[#0c100c] p-4">
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-semibold text-white">SceneDNA Investigation Pipeline</h3>
                <span className="ml-auto flex items-center gap-1 text-[11px] text-lime-300">
                  <span className="h-1.5 w-1.5 rounded-full bg-lime-400" /> {health?.clickhouse && health?.gemini ? "All Systems Online" : "Connecting"}
                </span>
              </div>
              <p className="text-xs text-slate-500">Scene Analyst → Pattern Analyst → Risk Analyst → Strategist</p>
              <div className="mt-3 space-y-2">
                {AGENTS.map((a) => (
                  <div key={a.key} className="flex items-center gap-3 rounded-xl border border-white/5 bg-white/[0.02] p-3">
                    <div className={`grid h-8 w-8 place-items-center rounded-full ${a.color}/20`}>
                      <Bot className={`h-4 w-4 ${a.color.replace("bg-", "text-")}`} />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-xs font-medium text-white">{a.name}</div>
                      <div className="truncate text-[10px] text-slate-500">{loading ? a.role : agentStats[a.key] ?? a.role}</div>
                    </div>
                    <Activity className={`h-3.5 w-3.5 ${loading ? "animate-pulse text-amber-300" : "text-lime-400"}`} />
                    <span className={`text-[10px] ${loading ? "text-amber-200" : "text-lime-300"}`}>{loading ? "Working" : "Done"}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-2xl border border-white/5 bg-[#0c100c] p-4">
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-semibold text-white">Live Agent Reasoning</h3>
                <button onClick={() => scrollTo("logs")} className="ml-auto text-[11px] text-slate-500 hover:text-white">View All →</button>
              </div>
              <div className="mt-3 max-h-96 space-y-3 overflow-auto pr-1">
                {loading && (
                  <div className="flex gap-2 text-[11px]">
                    <span className="font-mono text-slate-500">{clock(new Date())}</span>
                    <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-lime-400 animate-pulse" />
                    <span className="text-lime-200">Pattern Analyst</span>
                    <span className="text-slate-400">Running ClickHouse queries over MCP…</span>
                  </div>
                )}
                {liveLog.length === 0 && !loading && <p className="text-xs text-slate-500">Waiting for the first analysis.</p>}
                {liveLog.map((l, i) => (
                  <div key={i} className="grid grid-cols-[38px_10px_1fr] gap-2 text-[11px]">
                    <span className="font-mono text-slate-500">{clock(new Date(l.at))}</span>
                    <span className={`mt-1.5 h-1.5 w-1.5 rounded-full ${l.agent.color}`} />
                    <div>
                      <span className={`${l.agent.color.replace("bg-", "text-")}`}>{l.agent.name}</span>
                      <p className="text-slate-400">{l.text}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-2xl border border-white/5 bg-[#0c100c] p-4">
              <h3 className="text-sm font-semibold text-white">Recent Analyses</h3>
              <div className="mt-3 space-y-2">
                {recent.length === 0 && <p className="text-xs text-slate-500">No analyses yet.</p>}
                {recent.map((r) => (
                  <button key={r.title + r.at} onClick={() => r.scene_id !== "upload" && fetchPreset(r.scene_id)} className="flex w-full items-center gap-3 rounded-xl border border-white/5 bg-white/[0.02] p-2.5 text-left hover:border-white/15">
                    <div className="h-9 w-12 rounded-md bg-gradient-to-br from-slate-700 to-slate-900" />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-xs text-white">{r.title}</div>
                      <div className="text-[10px] text-slate-500">{ago(r.at)}</div>
                    </div>
                    <span className={`rounded-md border px-2 py-0.5 text-[10px] ${riskTone[r.risk]}`}>{r.risk} Risk</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="relative overflow-hidden rounded-2xl border border-white/5 bg-[#0c100c] p-5">
              <div className="pointer-events-none absolute -bottom-10 -right-10 h-40 w-40 rounded-full bg-[radial-gradient(circle,rgba(163,230,53,0.35),transparent_60%)]" />
              <p className="text-[11px] uppercase leading-relaxed tracking-[0.3em] text-slate-300">
                Great stories aren't just told.
                <br />
                They're understood.
              </p>
              <p className="mt-6 text-[10px] uppercase tracking-[0.3em] text-lime-300">SceneDNA</p>
            </div>
          </aside>
        </main>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------------------------
// Pieces
// ------------------------------------------------------------------------------------
function uniqueScenes(points: ScatterPoint[], top: Segment | null): ScatterPoint[] {
  const map = new Map<string, ScatterPoint>();
  for (const p of points) {
    if (top && (p.age_group !== top.age_group || p.device !== top.device)) continue;
    if (!map.has(p.scene_id)) map.set(p.scene_id, p);
  }
  if (map.size === 0) for (const p of points) if (!map.has(p.scene_id)) map.set(p.scene_id, p);
  return Array.from(map.values()).sort((a, b) => a.scene_id.localeCompare(b.scene_id));
}

function Kpi({ icon, value, label, spark, tone }: { icon: React.ReactNode; value: string; label: string; spark: number[]; tone: "lime" | "sky" | "violet" }) {
  const color = { lime: "text-lime-300 stroke-lime-400", sky: "text-sky-300 stroke-sky-400", violet: "text-violet-300 stroke-violet-400" }[tone];
  const max = Math.max(...spark);
  const pts = spark.map((v, i) => `${(i / (spark.length - 1)) * 100},${30 - (v / max) * 28}`).join(" ");
  return (
    <div className="flex items-center gap-4 rounded-2xl border border-white/5 bg-[#0c100c] p-4">
      <div className={`grid h-11 w-11 place-items-center rounded-xl bg-white/5 ${color.split(" ")[0]}`}>{icon}</div>
      <div className="flex-1">
        <div className="font-mono text-2xl text-white">{value}</div>
        <div className="text-xs text-slate-500">{label}</div>
      </div>
      <svg viewBox="0 0 100 32" className="h-8 w-20 overflow-visible">
        <polyline points={pts} fill="none" strokeWidth="2" className={color.split(" ")[1]} strokeLinejoin="round" strokeLinecap="round" />
      </svg>
    </div>
  );
}

function Feature({ label, pct, value }: { label: string; pct: number; value: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-28 text-slate-400">{label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded bg-white/10">
        <div className="h-full rounded bg-lime-400" style={{ width: `${Math.max(3, Math.min(100, pct))}%` }} />
      </div>
      <span className="w-14 text-right font-mono text-slate-200">{value}</span>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-28 text-slate-400">{label}</span>
      <span className="flex-1 truncate font-mono text-slate-200">{value}</span>
    </div>
  );
}

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <span className="rounded-md border border-white/10 bg-white/[0.03] px-2 py-1 text-slate-400">
      {label}: <span className="text-white">{value}</span>
    </span>
  );
}

function Scatter({ points, highlight }: { points: ScatterPoint[]; highlight: Segment | null }) {
  const W = 320;
  const H = 170;
  const padL = 34;
  const padB = 24;
  const maxX = Math.max(360, ...points.map((p) => p.duration_sec)) * 1.05;
  const maxY = Math.max(30, ...points.map((p) => p.exit_rate)) * 1.15;
  const x = (d: number) => padL + (d / maxX) * (W - padL - 8);
  const y = (r: number) => H - padB - (r / maxY) * (H - padB - 8);
  const corr = useMemo(() => {
    if (points.length < 3) return 0;
    const xs = points.map((p) => p.duration_sec);
    const ys = points.map((p) => p.exit_rate);
    const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
    const my = ys.reduce((a, b) => a + b, 0) / ys.length;
    const num = xs.reduce((a, v, i) => a + (v - mx) * (ys[i] - my), 0);
    const den = Math.sqrt(xs.reduce((a, v) => a + (v - mx) ** 2, 0) * ys.reduce((a, v) => a + (v - my) ** 2, 0));
    return den ? num / den : 0;
  }, [points]);
  return (
    <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
      <div className="mb-2 flex items-center text-xs">
        <span className="font-semibold text-white">Scene Duration vs. Exit Rate</span>
        <span className="ml-auto font-mono text-slate-500">r = {corr.toFixed(2)}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-44 w-full">
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line x1={padL} x2={W - 8} y1={y(f * maxY)} y2={y(f * maxY)} stroke="rgba(255,255,255,0.06)" />
            <text x={padL - 4} y={y(f * maxY) + 3} textAnchor="end" fontSize="8" fill="#64748b">{Math.round(f * maxY)}%</text>
          </g>
        ))}
        {[60, 120, 180, 240, 300, 360].map((d) => (
          <text key={d} x={x(d)} y={H - padB + 12} textAnchor="middle" fontSize="8" fill="#64748b">{d}</text>
        ))}
        <text x={(W + padL) / 2} y={H - 2} textAnchor="middle" fontSize="8" fill="#64748b">Scene duration (seconds)</text>
        {points.map((p, i) => {
          const hot = highlight && p.age_group === highlight.age_group && p.device === highlight.device;
          const level = p.exit_rate >= 18 ? "#fb923c" : p.exit_rate >= 11 ? "#fbbf24" : "#38bdf8";
          return <circle key={i} cx={x(p.duration_sec) + ((i % 7) - 3) * 1.6} cy={y(p.exit_rate)} r={hot ? 4 : 2.4} fill={level} fillOpacity={hot ? 0.95 : 0.55} stroke={hot ? "#fff" : "none"} strokeWidth={0.6}>
            <title>{`${p.scene_title} · ${p.age_group} ${p.device} · ${p.exit_rate}% exit`}</title>
          </circle>;
        })}
      </svg>
      <div className="mt-1 flex gap-4 text-[10px] text-slate-500">
        <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-sky-400" /> Low Risk</span>
        <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-amber-400" /> Medium Risk</span>
        <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-orange-400" /> High Risk</span>
      </div>
    </div>
  );
}

const TYPE_COLORS = ["#fbbf24", "#38bdf8", "#a3e635", "#f97316", "#a78bfa", "#64748b"];

function Donut({ types }: { types: SceneType[] }) {
  const total = types.reduce((a, t) => a + t.drops, 0) || 1;
  const views = types.reduce((a, t) => a + t.views, 0) || 1;
  const drops = types.reduce((a, t) => a + t.drops, 0);
  let acc = 0;
  const R = 40;
  const C = 2 * Math.PI * R;
  return (
    <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
      <p className="mb-2 text-xs font-semibold text-white">Drop-off by Scene Type</p>
      <div className="flex items-center gap-4">
        <svg viewBox="0 0 100 100" className="h-32 w-32 -rotate-90">
          <circle cx="50" cy="50" r={R} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="14" />
          {types.map((t, i) => {
            const frac = t.drops / total;
            const el = <circle key={t.scene_type} cx="50" cy="50" r={R} fill="none" stroke={TYPE_COLORS[i % TYPE_COLORS.length]} strokeWidth="14" strokeDasharray={`${frac * C} ${C}`} strokeDashoffset={-acc * C} />;
            acc += frac;
            return el;
          })}
          <text x="50" y="50" transform="rotate(90 50 50)" textAnchor="middle" fontSize="13" fill="#fff" fontWeight="600">{((drops / views) * 100).toFixed(0)}%</text>
          <text x="50" y="61" transform="rotate(90 50 50)" textAnchor="middle" fontSize="6" fill="#94a3b8">Drop-off Rate</text>
        </svg>
        <ul className="flex-1 space-y-1 text-[11px]">
          {types.map((t, i) => (
            <li key={t.scene_type} className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-sm" style={{ background: TYPE_COLORS[i % TYPE_COLORS.length] }} />
              <span className="flex-1 truncate text-slate-300">{t.scene_type.replace(/_/g, " ")}</span>
              <span className="font-mono text-slate-400">{Math.round((t.drops / total) * 100)}%</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

// keep the icon import list honest for tree-shaking
void BarChart3;
