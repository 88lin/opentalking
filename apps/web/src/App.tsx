import { useCallback, useEffect, useRef, useState } from "react";
import { ChatInput } from "./components/ChatInput";
import { ChatMessages } from "./components/ChatMessages";
import { SettingsPanel } from "./components/SettingsPanel";
import { StartOverlay } from "./components/StartOverlay";
import { SubtitleOverlay } from "./components/SubtitleOverlay";
import { TopBar } from "./components/TopBar";
import { VideoBackground } from "./components/VideoBackground";
import {
  apiDelete,
  apiGet,
  apiPost,
  buildApiUrl,
  type AvatarSummary,
  type CreateSessionResponse,
} from "./lib/api";
import { connectSse } from "./lib/sse";
import { startPlayback } from "./lib/webrtc";
import type { ConnectionStatus, Message } from "./types";

const MESSAGE_STORAGE_KEY = "opentalking-chat-history";

let msgCounter = 0;
function makeId() {
  return `msg-${++msgCounter}-${Date.now()}`;
}

function pickInitialAvatar(
  avatars: AvatarSummary[],
  registeredModels: string[],
): AvatarSummary | null {
  if (!avatars.length) return null;
  const available = new Set(registeredModels);
  // Prefer flashtalk, then musetalk, then any available
  return (
    avatars.find((a) => a.model_type === "flashtalk" && available.has("flashtalk")) ??
    avatars.find((a) => a.model_type === "musetalk" && available.has("musetalk")) ??
    avatars.find((a) => available.has(a.model_type)) ??
    avatars[0]
  );
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const subtitleAccRef = useRef("");

  // Data
  const [avatars, setAvatars] = useState<AvatarSummary[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [avatarId, setAvatarId] = useState("demo-avatar");
  const [model, setModel] = useState("wav2lip");

  // Connection
  const [connection, setConnection] = useState<ConnectionStatus>("idle");
  const [sessionId, setSessionId] = useState<string | null>(null);

  // Chat
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentSubtitle, setCurrentSubtitle] = useState("");
  const [isSpeaking, setIsSpeaking] = useState(false);

  // UI
  const [settingsOpen, setSettingsOpen] = useState(false);

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(MESSAGE_STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as Message[];
      if (!Array.isArray(parsed)) return;
      setMessages(parsed);
      msgCounter = Math.max(msgCounter, parsed.length);
    } catch (error) {
      console.warn("Failed to restore chat history", error);
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(MESSAGE_STORAGE_KEY, JSON.stringify(messages));
    } catch (error) {
      console.warn("Failed to persist chat history", error);
    }
  }, [messages]);

  const closePeerConnection = useCallback(() => {
    if (pcRef.current) {
      pcRef.current.close();
      pcRef.current = null;
    }
  }, []);

  const releaseSession = useCallback(async (sid: string, keepalive = false) => {
    try {
      await apiDelete(`/sessions/${sid}`, { keepalive });
    } catch (error) {
      console.warn("Failed to release session", sid, error);
    }
  }, []);

  const resetLiveState = useCallback(
    (clearMessages = false) => {
      closePeerConnection();
      setSessionId(null);
      setIsSpeaking(false);
      setCurrentSubtitle("");
      subtitleAccRef.current = "";
      if (clearMessages) {
        setMessages([]);
      }
    },
    [closePeerConnection],
  );

  // ---------- Init: fetch avatars & models ----------
  useEffect(() => {
    void (async () => {
      try {
        const [av, mo] = await Promise.all([
          apiGet<AvatarSummary[]>("/avatars"),
          apiGet<{ models: string[] }>("/models"),
        ]);
        setAvatars(av);
        setModels(mo.models);
        const initialAvatar = pickInitialAvatar(av, mo.models);
        if (initialAvatar) {
          setAvatarId(initialAvatar.id);
          setModel(initialAvatar.model_type);
        }
      } catch {
        setConnection("error");
      }
    })();
  }, []);

  // Keep model aligned with selected avatar
  useEffect(() => {
    const a = avatars.find((x) => x.id === avatarId);
    if (a) {
      setModel(a.model_type);
    }
  }, [avatarId, avatars]);

  // ---------- SSE ----------
  useEffect(() => {
    if (!sessionId) return;
    const stop = connectSse(buildApiUrl(`/sessions/${sessionId}/events`), (ev, data) => {
      if (ev === "speech.started") {
        setIsSpeaking(true);
        subtitleAccRef.current = "";
        setCurrentSubtitle("");
      }
      if (ev === "subtitle.chunk" && data && typeof data === "object") {
        const t = (data as { text?: string }).text;
        if (t) {
          subtitleAccRef.current = t;
          setCurrentSubtitle(t);
        }
      }
      if (ev === "speech.ended") {
        setIsSpeaking(false);
        const finalText = subtitleAccRef.current;
        if (finalText) {
          setMessages((prev) => [
            ...prev,
            { id: makeId(), role: "assistant", text: finalText, timestamp: Date.now() },
          ]);
        }
        setCurrentSubtitle("");
        subtitleAccRef.current = "";
      }
    });
    return stop;
  }, [sessionId]);

  // ---------- Actions ----------
  const handleStart = useCallback(async () => {
    if (!videoRef.current) return;

    const previousSessionId = sessionIdRef.current;
    if (previousSessionId) {
      await releaseSession(previousSessionId);
      resetLiveState();
    }

    setConnection("connecting");
    let createdSessionId: string | null = null;
    try {
      const created = await apiPost<CreateSessionResponse>("/sessions", {
        avatar_id: avatarId,
        model,
      });
      createdSessionId = created.session_id;
      setSessionId(created.session_id);

      closePeerConnection();
      const pc = await startPlayback(created.session_id, videoRef.current);
      pcRef.current = pc;
      // Unmute after user gesture so audio plays (autoplay policy requires muted initially)
      videoRef.current.muted = false;
      setConnection("live");
      await apiPost(`/sessions/${created.session_id}/start`, {});
    } catch (error) {
      if (createdSessionId) {
        await releaseSession(createdSessionId);
      }
      resetLiveState();
      console.warn("Failed to start session", error);
      setConnection("error");
    }
  }, [avatarId, closePeerConnection, model, releaseSession, resetLiveState]);

  const handleSend = useCallback(
    (text: string) => {
      if (!sessionId || !text) return;
      setMessages((prev) => [
        ...prev,
        { id: makeId(), role: "user", text, timestamp: Date.now() },
      ]);
      void apiPost(`/sessions/${sessionId}/speak`, { text }).catch(() => {
        setConnection("error");
      });
    },
    [sessionId],
  );

  const handleInterrupt = useCallback(() => {
    if (!sessionId) return;
    void apiPost(`/sessions/${sessionId}/interrupt`, {}).catch(() => {});
  }, [sessionId]);

  const handleAvatarChange = useCallback(
    (newId: string) => {
      setAvatarId(newId);
      void (async () => {
        const sid = sessionIdRef.current;
        if (sid) {
          await releaseSession(sid);
        }
        resetLiveState(true);
        setConnection("idle");
      })();
    },
    [releaseSession, resetLiveState],
  );

  const handleModelChange = useCallback((newModel: string) => {
    setModel(newModel);
    void (async () => {
      const sid = sessionIdRef.current;
      if (sid) {
        await releaseSession(sid);
      }
      resetLiveState();
      setConnection("idle");
    })();
  }, [releaseSession, resetLiveState]);

  useEffect(() => {
    const handlePageHide = () => {
      const sid = sessionIdRef.current;
      if (sid) {
        void releaseSession(sid, true);
      }
      closePeerConnection();
    };

    window.addEventListener("pagehide", handlePageHide);
    return () => window.removeEventListener("pagehide", handlePageHide);
  }, [closePeerConnection, releaseSession]);

  useEffect(() => {
    return () => {
      const sid = sessionIdRef.current;
      if (sid) {
        void releaseSession(sid, true);
      }
      closePeerConnection();
    };
  }, [closePeerConnection, releaseSession]);

  const currentAvatar = avatars.find((a) => a.id === avatarId) ?? null;
  const showStart = connection === "idle" || connection === "error";

  return (
    <>
      {/* Layer 0: Full-screen video background */}
      <VideoBackground ref={videoRef} />

      {/* Layer 1: Bottom gradient overlay */}
      <div
        className="pointer-events-none fixed inset-x-0 bottom-0 z-10"
        style={{
          height: "45vh",
          background: "linear-gradient(to top, rgba(0,0,0,0.75) 0%, transparent 100%)",
        }}
      />

      {/* Layer 2: Subtitle */}
      <SubtitleOverlay text={currentSubtitle} />

      {/* Layer 2: Chat messages */}
      <ChatMessages messages={messages} />

      {/* Layer 3: Top bar */}
      <TopBar
        connection={connection}
        onSettingsClick={() => setSettingsOpen(true)}
      />

      {/* Layer 3: Input bar */}
      <ChatInput
        onSend={handleSend}
        onInterrupt={handleInterrupt}
        isSpeaking={isSpeaking}
        disabled={connection !== "live"}
      />

      {/* Layer 4: Start overlay */}
      <StartOverlay
        avatar={currentAvatar}
        loading={connection === "connecting"}
        onStart={() => void handleStart()}
        visible={showStart}
      />

      {/* Layer 5: Settings panel */}
      <SettingsPanel
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        avatars={avatars}
        models={models.length ? models : ["wav2lip", "musetalk"]}
        avatarId={avatarId}
        model={model}
        onAvatarChange={handleAvatarChange}
        onModelChange={handleModelChange}
      />
    </>
  );
}
