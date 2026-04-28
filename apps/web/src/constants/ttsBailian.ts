/**
 * 百炼控制台多种语音合成路由（DashScope）。
 *
 * @see https://help.aliyun.com/zh/model-studio/cosyvoice-python-sdk
 */

/** CosyVoice（HTTP/SSE）；音色需与所选模型版本匹配，表内为示例 */
export const COSYVOICE_MODEL_OPTIONS: { id: string; label: string }[] = [
  { id: "cosyvoice-v3-flash", label: "CosyVoice v3 flash" },
  { id: "cosyvoice-v3-plus", label: "CosyVoice v3 plus" },
];

export const COSYVOICE_VOICE_OPTIONS: { id: string; label: string }[] = [
  { id: "longanyang", label: "longanyang（示例·男）" },
];

/** Sambert 经典链路；音色由模型名体现，不设独立 voice 字段 */
export const SAMBERT_MODEL_OPTIONS: { id: string; label: string }[] = [
  { id: "sambert-zhichu-v1", label: "sambert-zhichu-v1（知楚）" },
];

export type TtsProviderExtended =
  | "edge"
  | "dashscope"
  | "cosyvoice"
  | "sambert";

export function isEdgeTts(p: string): boolean {
  return p === "edge";
}
