import { useEffect, useState, type ReactNode } from "react";
import type { AvatarSummary } from "../lib/api";
import type { TtsProviderExtended } from "../constants/ttsBailian";

type VoiceOpt = { id: string; label: string; targetModel?: string | null };

export const SETTINGS_DOCK_EXPANDED_KEY = "opentalking-settings-dock-expanded";

const MODEL_LABELS: Record<string, string> = {
  flashhead: "FlashHead",
  flashtalk: "FlashTalk",
  musetalk: "MuseTalk",
  qingyu_v3: "Qingyu V3",
  wav2lip: "Wav2Lip",
};

interface SettingsPanelProps {
  /** 展开时显示表单；收起时仅保留右侧竖条入口 */
  expanded: boolean;
  onExpandedChange: (expanded: boolean) => void;
  avatars: AvatarSummary[];
  models: string[];
  avatarId: string;
  model: string;
  onAvatarChange: (id: string) => void;
  onModelChange: (m: string) => void;
  edgeVoice: string;
  onEdgeVoiceChange: (voiceId: string) => void;
  edgeVoiceOptions: { id: string; label: string }[];
  ttsProvider: TtsProviderExtended;
  onTtsProviderChange: (provider: TtsProviderExtended) => void;
  qwenModel: string;
  onQwenModelChange: (modelId: string) => void;
  qwenModelOptions: { id: string; label: string }[];
  qwenVoice: string;
  onQwenVoiceChange: (voiceId: string) => void;
  qwenVoiceOptions: VoiceOpt[];
  llmSystemPrompt: string;
  onLlmSystemPromptChange: (value: string) => void;
  onReferenceImageChange: (file: File | null) => void;
  onSavePrompt: () => void;
  onSaveReferenceImage: () => void;
  promptSaving?: boolean;
  referenceSaving?: boolean;
  onOpenVoiceClone?: () => void;
}

type SettingsSectionProps = {
  id: string;
  title: string;
  action?: ReactNode;
  children: ReactNode;
  open: boolean;
  onToggle: (id: string) => void;
};

function SettingsSection({ id, title, action, children, open, onToggle }: SettingsSectionProps) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white shadow-sm shadow-slate-200/40">
      <div className="flex items-center justify-between gap-2 border-b border-slate-100 px-4 py-3">
        <button
          type="button"
          onClick={() => onToggle(id)}
          className="flex min-w-0 flex-1 items-center gap-2 text-left"
          aria-expanded={open}
          aria-controls={`settings-section-${id}`}
        >
          <svg
            className={`h-3.5 w-3.5 shrink-0 text-slate-400 transition-transform ${open ? "rotate-90" : ""}`}
            viewBox="0 0 20 20"
            fill="currentColor"
            aria-hidden
          >
            <path d="M7 4l6 6-6 6V4z" />
          </svg>
          <h3 className="truncate text-sm font-semibold text-slate-900">{title}</h3>
        </button>
        {action}
      </div>
      {open ? (
        <div id={`settings-section-${id}`} className="p-4">
          {children}
        </div>
      ) : null}
    </section>
  );
}

export function SettingsPanel({
  expanded,
  onExpandedChange,
  avatars,
  models,
  avatarId,
  model,
  onAvatarChange,
  onModelChange,
  edgeVoice,
  onEdgeVoiceChange,
  edgeVoiceOptions,
  ttsProvider,
  onTtsProviderChange,
  qwenModel,
  onQwenModelChange,
  qwenModelOptions,
  qwenVoice,
  onQwenVoiceChange,
  qwenVoiceOptions,
  llmSystemPrompt,
  onLlmSystemPromptChange,
  onReferenceImageChange,
  onSavePrompt,
  onSaveReferenceImage,
  promptSaving = false,
  referenceSaving = false,
  onOpenVoiceClone,
}: SettingsPanelProps) {
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    avatars: true,
    model: true,
    voice: true,
    reference: true,
  });

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape" && expanded) {
        onExpandedChange(false);
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [expanded, onExpandedChange]);

  const toggleSection = (id: string) => {
    setOpenSections((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <aside className="flex min-h-0 flex-col border-r border-slate-200 bg-slate-50/70 lg:h-full lg:w-[280px] lg:shrink-0 lg:overflow-hidden">
      <div className="shrink-0 p-4 pb-3">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs font-medium text-slate-500">当前工作流</p>
            <h2 className="text-lg font-semibold text-slate-950">实时对话</h2>
          </div>
          <span className="rounded-full border border-cyan-200 bg-cyan-50 px-2 py-0.5 text-xs font-medium text-cyan-700">
            Studio
          </span>
        </div>
      </div>

      <div className="space-y-4 p-4 pt-0 lg:min-h-0 lg:flex-1 lg:overflow-y-auto">
        <SettingsSection
          id="avatars"
          title="数字人形象"
          open={openSections.avatars}
          onToggle={toggleSection}
          action={<span className="shrink-0 text-xs font-medium text-cyan-700">资产库</span>}
        >
          <div className="space-y-2">
            {avatars.length === 0 ? (
              <p className="rounded-lg border border-dashed border-slate-200 bg-slate-50 p-3 text-xs text-slate-500">
                正在读取数字人资产...
              </p>
            ) : (
              avatars.map((a) => (
                <button
                  key={a.id}
                  type="button"
                  onClick={() => onAvatarChange(a.id)}
                  className={`flex w-full items-center gap-3 rounded-lg border px-3 py-2 text-left transition ${
                    a.id === avatarId
                      ? "border-cyan-300 bg-cyan-50 shadow-sm"
                      : "border-slate-200 bg-white hover:border-slate-300"
                  }`}
                >
                  <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-slate-950 text-xs font-semibold text-white">
                    {(a.name ?? a.id).charAt(0).toUpperCase()}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-medium text-slate-900">{a.name ?? a.id}</span>
                    <span className="block truncate text-xs text-slate-500">{a.model_type}</span>
                  </span>
                </button>
              ))
            )}
          </div>
        </SettingsSection>

        <SettingsSection
          id="model"
          title="驱动模型"
          open={openSections.model}
          onToggle={toggleSection}
        >
          <select
            value={model}
            onChange={(e) => onModelChange(e.target.value)}
            className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm font-medium text-slate-800 outline-none transition focus:border-cyan-300 focus:bg-white"
          >
            {models.map((m) => (
              <option key={m} value={m}>
                {MODEL_LABELS[m] ?? m}
              </option>
            ))}
          </select>
        </SettingsSection>

        <SettingsSection
          id="voice"
          title="声音与角色"
          open={openSections.voice}
          onToggle={toggleSection}
          action={
            onOpenVoiceClone ? (
              <button
                type="button"
                onClick={() => onOpenVoiceClone()}
                className="shrink-0 text-xs font-medium text-cyan-700 hover:text-cyan-600"
              >
                复刻音色
              </button>
            ) : null
          }
        >
          <div className="space-y-3">
            <label className="block">
              <span className="mb-1.5 block text-xs text-slate-500">合成线路</span>
              <select
                value={ttsProvider}
                onChange={(e) => onTtsProviderChange(e.target.value as TtsProviderExtended)}
                className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-800 outline-none transition focus:border-cyan-300 focus:bg-white"
              >
                <option value="edge">微软 Edge（Neural）</option>
                <option value="dashscope">百炼 Qwen-TTS Realtime</option>
                <option value="cosyvoice">百炼 CosyVoice</option>
                <option value="sambert">百炼 Sambert</option>
              </select>
            </label>

            {ttsProvider === "edge" ? (
              <label className="block">
                <span className="mb-1.5 block text-xs text-slate-500">朗读音色</span>
                <select
                  value={edgeVoice}
                  onChange={(e) => onEdgeVoiceChange(e.target.value)}
                  className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-800 outline-none transition focus:border-cyan-300 focus:bg-white"
                >
                  {edgeVoiceOptions.map((o) => (
                    <option key={o.id} value={o.id}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </label>
            ) : (
              <>
                <label className="block">
                  <span className="mb-1.5 block text-xs text-slate-500">模型</span>
                  <select
                    value={qwenModel}
                    onChange={(e) => onQwenModelChange(e.target.value)}
                    className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-800 outline-none transition focus:border-cyan-300 focus:bg-white"
                  >
                    {qwenModelOptions.map((o) => (
                      <option key={o.id} value={o.id}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                </label>
                {qwenVoiceOptions.length > 0 ? (
                  <label className="block">
                    <span className="mb-1.5 block text-xs text-slate-500">音色</span>
                    <select
                      value={qwenVoice}
                      onChange={(e) => onQwenVoiceChange(e.target.value)}
                      className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-800 outline-none transition focus:border-cyan-300 focus:bg-white"
                    >
                      {qwenVoiceOptions.map((o) => (
                        <option key={o.id} value={o.id}>
                          {o.label}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
              </>
            )}

            <label className="block">
              <span className="mb-1.5 block text-xs text-slate-500">LLM System Prompt</span>
              <textarea
                value={llmSystemPrompt}
                onChange={(e) => onLlmSystemPromptChange(e.target.value)}
                rows={4}
                className="w-full resize-none rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm text-slate-800 outline-none transition focus:border-cyan-300 focus:bg-white"
                placeholder="输入新的系统提示词"
              />
            </label>
            <button
              type="button"
              onClick={onSavePrompt}
              disabled={promptSaving}
              className="w-full rounded-lg bg-slate-950 px-3 py-2.5 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {promptSaving ? "保存中..." : "保存 Prompt"}
            </button>
          </div>
        </SettingsSection>

        <SettingsSection
          id="reference"
          title="参考图"
          open={openSections.reference}
          onToggle={toggleSection}
        >
          <div className="space-y-3">
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600 file:mr-3 file:rounded-md file:border-0 file:bg-cyan-600 file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-white hover:file:bg-cyan-500"
              onChange={(e) => onReferenceImageChange(e.target.files?.[0] ?? null)}
            />
            <button
              type="button"
              onClick={onSaveReferenceImage}
              disabled={referenceSaving}
              className="w-full rounded-lg border border-cyan-200 bg-cyan-50 px-3 py-2.5 text-sm font-semibold text-cyan-700 transition hover:bg-cyan-100 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {referenceSaving ? "上传中..." : "上传参考图"}
            </button>
          </div>
        </SettingsSection>
      </div>
    </aside>
  );
}
