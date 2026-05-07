import { buildTTSPreviewPayload } from "./ttsPreview";

const qwenPayload = buildTTSPreviewPayload({
  text: "  你好  ",
  voice: "voice-clone-1",
  provider: "dashscope",
  model: "qwen3-tts-flash-realtime",
});

qwenPayload satisfies {
  text: string;
  voice?: string;
  tts_provider: "dashscope" | "edge" | "cosyvoice" | "sambert";
  tts_model?: string;
};

const edgePayload = buildTTSPreviewPayload({
  text: "你好",
  voice: "zh-CN-XiaoxiaoNeural",
  provider: "edge",
  model: "",
});

edgePayload satisfies { tts_model?: string };
