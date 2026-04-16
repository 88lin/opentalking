import { useState, useCallback, type KeyboardEvent } from "react";

interface ChatInputProps {
  onSend: (text: string) => void;
  onInterrupt: () => void;
  isSpeaking: boolean;
  disabled: boolean;
}

export function ChatInput({ onSend, onInterrupt, isSpeaking, disabled }: ChatInputProps) {
  const [text, setText] = useState("");

  const handleSend = useCallback(() => {
    const trimmed = text.trim();
    if (!trimmed) return;
    onSend(trimmed);
    setText("");
  }, [text, onSend]);

  const handleKey = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  return (
    <div className="glass fixed inset-x-0 bottom-0 z-30 px-4 pb-[env(safe-area-inset-bottom,8px)] pt-3">
      <div className="mx-auto flex max-w-2xl items-end gap-2">
        <textarea
          className="flex-1 resize-none rounded-3xl bg-white/10 px-5 py-3 text-sm text-slate-100 placeholder-slate-500 outline-none transition-colors focus:bg-white/15"
          placeholder="输入消息..."
          rows={1}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKey}
          disabled={disabled}
        />

        {isSpeaking ? (
          <button
            type="button"
            onClick={onInterrupt}
            className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-red-500 text-white transition-colors hover:bg-red-600"
            title="停止"
          >
            {/* Stop icon */}
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 16 16" fill="currentColor">
              <rect x="3" y="3" width="10" height="10" rx="1" />
            </svg>
          </button>
        ) : (
          <button
            type="button"
            onClick={handleSend}
            disabled={disabled || !text.trim()}
            className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-cyan-500 text-white transition-colors hover:bg-cyan-600 disabled:opacity-40"
            title="发送"
          >
            {/* Send arrow icon */}
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 10.5 12 3m0 0 7.5 7.5M12 3v18" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
}
