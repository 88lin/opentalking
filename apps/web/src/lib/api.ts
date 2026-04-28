export const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

export function buildApiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

export function buildApiDownloadUrl(path: string): string {
  return buildApiUrl(path);
}

/** WebSocket：相对 ``/api`` 走当前页 host；绝对 ``VITE_API_BASE`` 时与 HTTP 同机（与主仓一致） */
export function buildWsUrl(path: string): string {
  const p = path.startsWith("/") ? path : `/${path}`;
  if (typeof window === "undefined") {
    return `ws://127.0.0.1:5173${API_BASE}${p}`;
  }
  try {
    const baseUrl = new URL(API_BASE, window.location.origin);
    const wsProto = baseUrl.protocol === "https:" ? "wss:" : "ws:";
    return `${wsProto}//${baseUrl.host}${baseUrl.pathname}${p}`;
  } catch {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}${API_BASE}${p}`;
  }
}

export async function apiGet<T>(path: string): Promise<T> {
  const r = await fetch(buildApiUrl(path));
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(buildApiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

/** multipart/form-data（语音识别 speak_audio / transcribe） */
export async function apiPostForm<T>(path: string, form: FormData, init?: RequestInit): Promise<T> {
  const r = await fetch(buildApiUrl(path), { method: "POST", body: form, ...init });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export async function apiDelete<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(buildApiUrl(path), { ...init, method: "DELETE" });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export type AvatarSummary = {
  id: string;
  name: string | null;
  model_type: string;
  width: number;
  height: number;
};

export type CreateSessionResponse = { session_id: string; status: string };

/** GET /voices 返回的音色目录项（含 SQLite 中的系统预设与复刻） */
export type VoiceCatalogItem = {
  id: number;
  user_id: number;
  provider: string;
  voice_id: string;
  display_label: string;
  target_model: string | null;
  source: "system" | "clone" | string;
};
