import { useEffect, useRef } from "react";
import type { AvatarSummary } from "../lib/api";

interface SettingsPanelProps {
  open: boolean;
  onClose: () => void;
  avatars: AvatarSummary[];
  models: string[];
  avatarId: string;
  model: string;
  onAvatarChange: (id: string) => void;
  onModelChange: (m: string) => void;
}

export function SettingsPanel({
  open,
  onClose,
  avatars,
  models,
  avatarId,
  model,
  onAvatarChange,
  onModelChange,
}: SettingsPanelProps) {
  const panelRef = useRef<HTMLDivElement>(null);

  // Close on click outside
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    // Delay to avoid triggering on the same click that opened the panel
    const timer = setTimeout(() => document.addEventListener("click", handler), 10);
    return () => {
      clearTimeout(timer);
      document.removeEventListener("click", handler);
    };
  }, [open, onClose]);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/40" />

      {/* Panel — right on desktop, bottom on mobile */}
      <div
        ref={panelRef}
        className="glass absolute bottom-0 right-0 top-0 w-80 animate-slide-in-right overflow-y-auto p-6 max-sm:inset-x-0 max-sm:top-auto max-sm:w-full max-sm:animate-slide-in-up max-sm:rounded-t-2xl"
      >
        {/* Header */}
        <div className="mb-6 flex items-center justify-between">
          <h3 className="text-base font-medium text-white">设置</h3>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1 text-slate-400 transition-colors hover:bg-white/10 hover:text-white"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Avatar selection */}
        <div className="mb-6">
          <label className="mb-2 block text-xs font-medium uppercase tracking-wider text-slate-400">
            Avatar
          </label>
          <div className="flex flex-col gap-2">
            {avatars.map((a) => (
              <button
                key={a.id}
                type="button"
                onClick={() => onAvatarChange(a.id)}
                className={`flex items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors ${
                  a.id === avatarId
                    ? "bg-cyan-500/20 text-white ring-1 ring-cyan-500/40"
                    : "text-slate-300 hover:bg-white/10"
                }`}
              >
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-white/10 text-xs text-slate-400">
                  {a.id.charAt(0).toUpperCase()}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm">{a.name ?? a.id}</div>
                  <div className="text-xs text-slate-500">{a.model_type}</div>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Model selection */}
        <div>
          <label className="mb-2 block text-xs font-medium uppercase tracking-wider text-slate-400">
            模型
          </label>
          <select
            value={model}
            onChange={(e) => onModelChange(e.target.value)}
            className="w-full rounded-xl bg-white/10 px-3 py-2.5 text-sm text-slate-100 outline-none transition-colors focus:bg-white/15"
          >
            {models.map((m) => (
              <option key={m} value={m} className="bg-slate-900">
                {m}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
}
