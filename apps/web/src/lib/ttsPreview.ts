import { apiPostBlob } from "./api";
import type { TtsProviderExtended } from "../constants/ttsBailian";

export const DEFAULT_TTS_PREVIEW_TEXT = "你好，我正在测试音色。";

export type TTSPreviewPayload = {
  text: string;
  voice?: string;
  tts_provider: TtsProviderExtended;
  tts_model?: string;
};

export function buildTTSPreviewPayload({
  text,
  voice,
  provider,
  model,
}: {
  text: string;
  voice: string;
  provider: TtsProviderExtended;
  model: string;
}): TTSPreviewPayload {
  const payload: TTSPreviewPayload = {
    text: text.trim(),
    tts_provider: provider,
  };
  const trimmedVoice = voice.trim();
  if (trimmedVoice && provider !== "sambert") {
    payload.voice = trimmedVoice;
  }
  if (provider !== "edge") {
    payload.tts_model = model;
  }
  return payload;
}

export async function requestTTSPreview(payload: TTSPreviewPayload): Promise<Blob> {
  return apiPostBlob("/tts/preview", payload);
}
