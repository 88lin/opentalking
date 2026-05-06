import { useCallback, useRef, useState } from "react";
import { apiPostForm } from "../lib/api";
import { COSYVOICE_MODEL_OPTIONS } from "../constants/ttsBailian";
import { QWEN_VOICE_CLONE_TARGET_OPTIONS } from "../constants/ttsQwen";

/** 固定朗读文案：用于复刻时与百炼侧要求一致 */
export const BAILIAN_CLONE_SAMPLE_TEXT =
  "大家好，这是一段用于音色复刻的固定文本。我会用自然、清晰、平稳的语速读完它。";

function pickRecorderMime(): string | undefined {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"];
  for (const t of candidates) {
    if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(t)) {
      return t;
    }
  }
  return undefined;
}

type CloneProvider = "dashscope" | "cosyvoice";

interface BailianVoiceCloneProps {
  onSuccess: () => void;
  onClose: () => void;
}

export function BailianVoiceClone({ onSuccess, onClose }: BailianVoiceCloneProps) {
  const [provider, setProvider] = useState<CloneProvider>("dashscope");
  const [targetModel, setTargetModel] = useState(
    () =>
      (provider === "dashscope"
        ? QWEN_VOICE_CLONE_TARGET_OPTIONS[0]?.id
        : COSYVOICE_MODEL_OPTIONS[0]?.id) ?? "",
  );
  const [displayLabel, setDisplayLabel] = useState("我的复刻音色");
  const [prefix, setPrefix] = useState("");
  const [preferredName, setPreferredName] = useState("");
  const [recording, setRecording] = useState(false);
  const [blob, setBlob] = useState<Blob | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const mrRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  const onProviderChange = (p: CloneProvider) => {
    setProvider(p);
    if (p === "dashscope") {
      setTargetModel(QWEN_VOICE_CLONE_TARGET_OPTIONS[0]?.id ?? "");
    } else {
      setTargetModel(COSYVOICE_MODEL_OPTIONS[0]?.id ?? "");
    }
  };

  const stopRecording = useCallback(async () => {
    const mr = mrRef.current;
    if (!mr || mr.state === "inactive") {
      mrRef.current = null;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      setRecording(false);
      return;
    }
    await new Promise<void>((resolve) => {
      mr.onstop = () => resolve();
      try {
        mr.stop();
      } catch {
        resolve();
      }
    });
    mrRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    const mime = mr.mimeType || "audio/webm";
    const b = new Blob(chunksRef.current, { type: mime });
    chunksRef.current = [];
    setBlob(b);
    setRecording(false);
  }, []);

  const startRecording = useCallback(async () => {
    setMessage(null);
    setBlob(null);
    chunksRef.current = [];
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef.current = stream;
    const mime = pickRecorderMime();
    const mr = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
    mrRef.current = mr;
    mr.ondataavailable = (ev) => {
      if (ev.data.size > 0) chunksRef.current.push(ev.data);
    };
    mr.start(200);
    setRecording(true);
  }, []);

  const submit = useCallback(async () => {
    if (!blob || blob.size < 64) {
      setMessage("请先录制一段音频");
      return;
    }
    if (!targetModel.trim()) {
      setMessage("请选择目标模型");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const ext = blob.type.includes("webm") ? "webm" : blob.type.includes("ogg") ? "ogg" : "webm";
      const fd = new FormData();
      fd.append("provider", provider);
      fd.append("target_model", targetModel.trim());
      fd.append("display_label", displayLabel.trim() || "我的复刻音色");
      fd.append("audio", blob, `sample.${ext}`);
      fd.append("prefix", prefix.trim());
      fd.append("preferred_name", preferredName.trim());
      const res = await apiPostForm<{
        ok?: boolean;
        message?: string;
        voice_id?: string;
      }>("/voices/clone", fd);
      setMessage(res.message ?? `已生成 voice_id：${res.voice_id ?? "?"}`);
      onSuccess();
    } catch (e) {
      setMessage(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [blob, displayLabel, onSuccess, preferredName, prefix, provider, targetModel]);

  return (
    <div className="mx-auto max-w-xl rounded-lg border border-slate-200 bg-white text-sm text-slate-800 shadow-sm shadow-slate-200/70">
      <div className="flex items-start justify-between gap-3 border-b border-slate-100 px-4 py-3">
        <div>
          <p className="text-xs font-medium text-slate-500">声音与角色</p>
          <h2 className="mt-0.5 text-base font-semibold text-slate-950">百炼音色复刻</h2>
        </div>
        <button
          type="button"
          className="rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-600 transition hover:bg-slate-50"
          onClick={onClose}
        >
          关闭
        </button>
      </div>

      <div className="space-y-4 p-4">
        <div>
          <p className="text-xs leading-relaxed text-slate-500">
            请朗读下方固定文案并录音。千问复刻走 base64，内网可用；CosyVoice 需本服务对公网可访问或配置{" "}
            <code className="rounded bg-slate-100 px-1 py-0.5 text-slate-700">OPENTALKING_PUBLIC_BASE_URL</code>。
          </p>
          <blockquote className="mt-3 rounded-lg border border-cyan-200 bg-cyan-50 px-3 py-2 text-xs leading-relaxed text-cyan-900">
            {BAILIAN_CLONE_SAMPLE_TEXT}
          </blockquote>
        </div>

        <div className="grid gap-3 text-xs sm:grid-cols-2">
          <label className="block">
            <span className="mb-1.5 block text-slate-500">线路</span>
            <select
              className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-800 outline-none transition focus:border-cyan-300 focus:bg-white"
              value={provider}
              onChange={(e) => onProviderChange(e.target.value as CloneProvider)}
              disabled={busy}
            >
              <option value="dashscope">千问（DashScope 复刻）</option>
              <option value="cosyvoice">CosyVoice</option>
            </select>
          </label>
          <label className="block">
            <span className="mb-1.5 block text-slate-500">目标模型</span>
            <select
              className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-800 outline-none transition focus:border-cyan-300 focus:bg-white"
              value={targetModel}
              onChange={(e) => setTargetModel(e.target.value)}
              disabled={busy}
            >
              {(provider === "dashscope" ? QWEN_VOICE_CLONE_TARGET_OPTIONS : COSYVOICE_MODEL_OPTIONS).map(
                (o) => (
                  <option key={o.id} value={o.id}>
                    {o.label}
                  </option>
                ),
              )}
            </select>
          </label>
        </div>

        <div className="grid gap-3 text-xs sm:grid-cols-2">
          <label className="block">
            <span className="mb-1.5 block text-slate-500">显示名称</span>
            <input
              className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-cyan-300 focus:bg-white"
              value={displayLabel}
              onChange={(e) => setDisplayLabel(e.target.value)}
              disabled={busy}
            />
          </label>
          {provider === "cosyvoice" ? (
            <label className="block">
              <span className="mb-1.5 block text-slate-500">前缀 prefix（可选，小写字母数字）</span>
              <input
                className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-cyan-300 focus:bg-white"
                value={prefix}
                onChange={(e) => setPrefix(e.target.value)}
                placeholder="留空则自动生成"
                disabled={busy}
              />
            </label>
          ) : (
            <label className="block">
              <span className="mb-1.5 block text-slate-500">preferred_name（可选，小写）</span>
              <input
                className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-cyan-300 focus:bg-white"
                value={preferredName}
                onChange={(e) => setPreferredName(e.target.value)}
                placeholder="留空则自动生成"
                disabled={busy}
              />
            </label>
          )}
        </div>

      <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
        <div className="flex flex-wrap items-center gap-2">
          {!recording ? (
            <button
              type="button"
              className="rounded-lg bg-emerald-600 px-4 py-2 text-xs font-semibold text-white transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
              disabled={busy}
              onClick={() => void startRecording()}
            >
              开始录音
            </button>
          ) : (
            <button
              type="button"
              className="rounded-lg bg-rose-600 px-4 py-2 text-xs font-semibold text-white transition hover:bg-rose-500"
              onClick={() => void stopRecording()}
            >
              停止
            </button>
          )}
          <button
            type="button"
            className="rounded-lg border border-cyan-200 bg-cyan-50 px-4 py-2 text-xs font-semibold text-cyan-700 transition hover:bg-cyan-100 disabled:cursor-not-allowed disabled:opacity-50"
            disabled={busy || !blob}
            onClick={() => void submit()}
          >
            {busy ? "提交中…" : "上传并复刻"}
          </button>
          {blob ? (
            <span className="text-xs text-slate-500">已录 {Math.round(blob.size / 1024)} KB</span>
          ) : null}
        </div>
      </div>
      {message ? (
        <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs leading-relaxed text-amber-800">
          {message}
        </p>
      ) : null}
      </div>
    </div>
  );
}
