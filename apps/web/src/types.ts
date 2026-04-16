export type ConnectionStatus = "idle" | "connecting" | "live" | "error";

export interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  timestamp: number;
}
