interface SubtitleOverlayProps {
  text: string;
}

export function SubtitleOverlay({ text }: SubtitleOverlayProps) {
  if (!text) return null;

  return (
    <div className="fixed inset-x-0 bottom-[45vh] z-20 flex justify-center px-4">
      <div className="glass animate-fade-in rounded-xl px-5 py-2.5 text-center text-sm leading-relaxed text-slate-100">
        {text}
      </div>
    </div>
  );
}
