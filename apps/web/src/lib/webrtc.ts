import { apiPost } from "./api";

export async function startPlayback(sessionId: string, videoEl: HTMLVideoElement) {
  const pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });
  const mediaStream = new MediaStream();
  videoEl.srcObject = mediaStream;

  pc.ontrack = (ev) => {
    const track = ev.track;
    if (!track) return;
    const hasTrack = mediaStream.getTracks().some((t) => t.id === track.id);
    if (!hasTrack) {
      mediaStream.addTrack(track);
    }
    videoEl.play().catch(() => {});
  };

  const cleanup = () => {
    videoEl.pause();
    videoEl.srcObject = null;
  };
  pc.addEventListener("connectionstatechange", () => {
    if (
      pc.connectionState === "closed"
      || pc.connectionState === "failed"
      || pc.connectionState === "disconnected"
    ) {
      cleanup();
    }
  });
  pc.addEventListener("iceconnectionstatechange", () => {
    if (
      pc.iceConnectionState === "closed"
      || pc.iceConnectionState === "failed"
      || pc.iceConnectionState === "disconnected"
    ) {
      cleanup();
    }
  });

  pc.addTransceiver("video", { direction: "recvonly" });
  pc.addTransceiver("audio", { direction: "recvonly" });

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const answer = await apiPost<{ sdp: string; type: RTCSdpType }>(
    `/sessions/${sessionId}/webrtc/offer`,
    { sdp: pc.localDescription?.sdp ?? "", type: pc.localDescription?.type ?? "offer" }
  );

  await pc.setRemoteDescription(new RTCSessionDescription(answer));
  return pc;
}
