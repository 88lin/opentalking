import type { Message } from "../types";

interface ChatBubbleProps {
  message: Message;
}

export function ChatBubble({ message }: ChatBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div
      className={`flex animate-slide-up ${isUser ? "justify-end" : "justify-start"}`}
    >
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed sm:max-w-[70%] ${
          isUser
            ? "bg-blue-500/80 text-white"
            : "bg-white/10 text-slate-100"
        }`}
      >
        {message.text}
      </div>
    </div>
  );
}
