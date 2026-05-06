import { forwardRef } from "react";

type VideoBackgroundProps = {
  className?: string;
};

export const VideoBackground = forwardRef<HTMLVideoElement, VideoBackgroundProps>(
  ({ className }, ref) => (
    <video
      ref={ref}
      className={className ?? "absolute inset-0 h-full w-full object-contain"}
      autoPlay
      playsInline
      muted
    />
  ),
);

VideoBackground.displayName = "VideoBackground";
