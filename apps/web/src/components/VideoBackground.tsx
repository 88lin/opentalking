import { forwardRef } from "react";

export const VideoBackground = forwardRef<HTMLVideoElement>((_props, ref) => (
  <video
    ref={ref}
    className="fixed inset-0 h-full w-full object-contain"
    autoPlay
    playsInline
    muted
    style={{ zIndex: 0 }}
  />
));

VideoBackground.displayName = "VideoBackground";
