export const DEFAULT_START_FRAME = 1;
export const DEFAULT_END_FRAME = 120;
export const DEFAULT_FPS = 30;
export const DEFAULT_ZOOM = 14;
export const MIN_ZOOM = 4;
export const MAX_ZOOM = 60;

export const SUMMARY_TRACK_COLOR = "#f59e0b";
export const DEFAULT_BONE_COLOR = "#38bdf8";
export const SELECTED_KEYFRAME_COLOR = "#fbbf24";

export const BONE_TRACK_COLORS = [
  "#38bdf8",
  "#a855f7",
  "#ec4899",
  "#10b981",
  "#f97316",
  "#eab308",
] as const;

export const CHANNEL_COLORS = {
  position: "#ef4444",
  rotation: "#3b82f6",
  scale: "#22c55e",
  property: "#10b981",
} as const;

export const FPS_OPTIONS = [24, 30, 60] as const;
export const PLAYBACK_SPEED_OPTIONS = [0.25, 0.5, 1, 1.5, 2] as const;
