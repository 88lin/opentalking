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

/** Rich error type so callers can show the FastAPI {"detail": "..."} message. */
export class ApiError extends Error {
  status: number;
  detail: string | null;
  body: string;
  constructor(status: number, detail: string | null, body: string) {
    super(detail || `HTTP ${status}: ${body}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.body = body;
  }
}

async function throwIfNotOk(r: Response): Promise<void> {
  if (r.ok) return;
  const body = await r.text();
  let detail: string | null = null;
  try {
    const parsed = JSON.parse(body);
    if (typeof parsed?.detail === "string") {
      detail = parsed.detail;
    } else if (Array.isArray(parsed?.detail)) {
      // FastAPI validation errors arrive as a list of {loc, msg, ...}
      detail = parsed.detail
        .map((d: { msg?: string }) => d?.msg ?? JSON.stringify(d))
        .join("; ");
    } else if (parsed?.detail != null) {
      detail = JSON.stringify(parsed.detail);
    }
  } catch {
    // body wasn't JSON; leave detail null
  }
  throw new ApiError(r.status, detail, body);
}

export async function apiGet<T>(path: string): Promise<T> {
  const r = await fetch(buildApiUrl(path));
  await throwIfNotOk(r);
  return r.json() as Promise<T>;
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(buildApiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  await throwIfNotOk(r);
  return r.json() as Promise<T>;
}

export async function apiPostBlob(path: string, body?: unknown): Promise<Blob> {
  const r = await fetch(buildApiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  await throwIfNotOk(r);
  return r.blob();
}

/** multipart/form-data（语音识别 speak_audio / transcribe） */
export async function apiPostForm<T>(path: string, form: FormData, init?: RequestInit): Promise<T> {
  const r = await fetch(buildApiUrl(path), { method: "POST", body: form, ...init });
  await throwIfNotOk(r);
  return r.json() as Promise<T>;
}

export async function apiDelete<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(buildApiUrl(path), { ...init, method: "DELETE" });
  await throwIfNotOk(r);
  return r.json() as Promise<T>;
}

export type AvatarSummary = {
  id: string;
  name: string | null;
  model_type: string;
  width: number;
  height: number;
  is_custom: boolean;
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
