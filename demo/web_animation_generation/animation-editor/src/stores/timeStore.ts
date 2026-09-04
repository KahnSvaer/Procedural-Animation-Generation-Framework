import { create } from "zustand";
import { Object3D, AnimationClip, Bone, SkinnedMesh } from "three";

export interface KeyframePoint {
  frame: number;
  value?: number | number[];
  interpolation?: "linear" | "bezier" | "constant";
  selected?: boolean;
}

export interface BoneTrackChannel {
  id: string;
  name: string;
  color: string;
  keyframes: KeyframePoint[];
}

export interface BoneTrack {
  id: string;
  name: string;
  type: "summary" | "bone" | "property";
  parentId?: string;
  color?: string;
  expanded?: boolean;
  muted?: boolean;
  locked?: boolean;
  keyframes: KeyframePoint[];
  channels?: BoneTrackChannel[];
}

interface TimeStore {
  currentFrame: number;
  startFrame: number;
  endFrame: number;
  fps: number;
  isPlaying: boolean;
  isLooping: boolean;
  playbackSpeed: number;

  zoom: number;
  scrollLeft: number;
  selectedBoneId: string | null;
  selectedKeyframe: { trackId: string; frame: number } | null;

  tracks: BoneTrack[];

  setCurrentFrame: (frame: number) => void;
  setStartFrame: (frame: number) => void;
  setEndFrame: (frame: number) => void;
  setFps: (fps: number) => void;
  setIsPlaying: (isPlaying: boolean) => void;
  togglePlay: () => void;
  setIsLooping: (isLooping: boolean) => void;
  setPlaybackSpeed: (speed: number) => void;

  stepForward: (step?: number) => void;
  stepBackward: (step?: number) => void;
  goToStart: () => void;
  goToEnd: () => void;

  setZoom: (zoom: number) => void;
  setScrollLeft: (scrollLeft: number) => void;
  setSelectedBoneId: (boneId: string | null) => void;
  setSelectedKeyframe: (keyframe: { trackId: string; frame: number } | null) => void;

  toggleTrackExpanded: (trackId: string) => void;
  toggleTrackMuted: (trackId: string) => void;
  toggleTrackLocked: (trackId: string) => void;
  addKeyframe: (trackId: string, frame: number, value?: number) => void;
  removeKeyframe: (trackId: string, frame: number) => void;
  populateFromGLTF: (model: Object3D, animations: AnimationClip[], fps?: number) => void;
  resetDefaultTracks: () => void;
}

const DEFAULT_TRACKS: BoneTrack[] = [
  {
    id: "summary",
    name: "Summary",
    type: "summary",
    color: "#f59e0b",
    expanded: true,
    keyframes: [],
  },
];

export const useTimeStore = create<TimeStore>((set, get) => ({
  currentFrame: 1,
  startFrame: 1,
  endFrame: 120,
  fps: 30,
  isPlaying: false,
  isLooping: true,
  playbackSpeed: 1,

  zoom: 14,
  scrollLeft: 0,
  selectedBoneId: null,
  selectedKeyframe: null,

  tracks: DEFAULT_TRACKS,

  setCurrentFrame: (frame) => {
    const { startFrame, endFrame } = get();
    const clamped = Math.max(startFrame, Math.min(endFrame, Math.round(frame)));
    set({ currentFrame: clamped });
  },

  setStartFrame: (startFrame) => {
    set((state) => ({
      startFrame,
      currentFrame: Math.max(startFrame, state.currentFrame),
    }));
  },

  setEndFrame: (endFrame) => {
    set((state) => ({
      endFrame,
      currentFrame: Math.min(endFrame, state.currentFrame),
    }));
  },

  setFps: (fps) => set({ fps }),
  setIsPlaying: (isPlaying) => set({ isPlaying }),
  togglePlay: () => set((state) => ({ isPlaying: !state.isPlaying })),
  setIsLooping: (isLooping) => set({ isLooping }),
  setPlaybackSpeed: (playbackSpeed) => set({ playbackSpeed }),

  stepForward: (step = 1) => {
    const { currentFrame, startFrame, endFrame, isLooping } = get();
    let next = currentFrame + step;
    if (next > endFrame) {
      next = isLooping ? startFrame : endFrame;
    }
    set({ currentFrame: next });
  },

  stepBackward: (step = 1) => {
    const { currentFrame, startFrame, endFrame, isLooping } = get();
    let prev = currentFrame - step;
    if (prev < startFrame) {
      prev = isLooping ? endFrame : startFrame;
    }
    set({ currentFrame: prev });
  },

  goToStart: () => set((state) => ({ currentFrame: state.startFrame })),
  goToEnd: () => set((state) => ({ currentFrame: state.endFrame })),

  setZoom: (zoom) => set({ zoom: Math.max(4, Math.min(60, zoom)) }),
  setScrollLeft: (scrollLeft) => set({ scrollLeft: Math.max(0, scrollLeft) }),
  setSelectedBoneId: (selectedBoneId) => set({ selectedBoneId }),
  setSelectedKeyframe: (selectedKeyframe) => set({ selectedKeyframe }),

  toggleTrackExpanded: (trackId) => {
    set((state) => ({
      tracks: state.tracks.map((track) =>
        track.id === trackId ? { ...track, expanded: !track.expanded } : track
      ),
    }));
  },

  toggleTrackMuted: (trackId) => {
    set((state) => ({
      tracks: state.tracks.map((track) =>
        track.id === trackId ? { ...track, muted: !track.muted } : track
      ),
    }));
  },

  toggleTrackLocked: (trackId) => {
    set((state) => ({
      tracks: state.tracks.map((track) =>
        track.id === trackId ? { ...track, locked: !track.locked } : track
      ),
    }));
  },

  addKeyframe: (trackId, frame, value) => {
    set((state) => {
      const tracks = state.tracks.map((track) => {
        if (track.id === trackId) {
          const exists = track.keyframes.some((k) => k.frame === frame);
          if (exists) {
            return {
              ...track,
              keyframes: track.keyframes.map((k) =>
                k.frame === frame ? { ...k, value } : k
              ),
            };
          }
          const nextKeyframes = [...track.keyframes, { frame, value }].sort(
            (a, b) => a.frame - b.frame
          );
          return { ...track, keyframes: nextKeyframes };
        }
        return track;
      });

      const allFrames = new Set<number>();
      tracks.forEach((t) => {
        if (t.id !== "summary") {
          t.keyframes.forEach((k) => allFrames.add(k.frame));
        }
      });
      const summaryKeyframes = Array.from(allFrames)
        .sort((a, b) => a - b)
        .map((f) => ({ frame: f }));

      return {
        tracks: tracks.map((t) =>
          t.id === "summary" ? { ...t, keyframes: summaryKeyframes } : t
        ),
      };
    });
  },

  removeKeyframe: (trackId, frame) => {
    set((state) => {
      const tracks = state.tracks.map((track) => {
        if (track.id === trackId) {
          return {
            ...track,
            keyframes: track.keyframes.filter((k) => k.frame !== frame),
          };
        }
        return track;
      });

      const allFrames = new Set<number>();
      tracks.forEach((t) => {
        if (t.id !== "summary") {
          t.keyframes.forEach((k) => allFrames.add(k.frame));
        }
      });
      const summaryKeyframes = Array.from(allFrames)
        .sort((a, b) => a - b)
        .map((f) => ({ frame: f }));

      return {
        tracks: tracks.map((t) =>
          t.id === "summary" ? { ...t, keyframes: summaryKeyframes } : t
        ),
      };
    });
  },

  populateFromGLTF: (model, animations, customFps) => {
    const fps = customFps || get().fps;
    const colors = ["#38bdf8", "#a855f7", "#ec4899", "#10b981", "#f97316", "#eab308"];

    const boneMap = new Map<string, Bone>();
    model.traverse((child) => {
      if ((child as Bone).isBone || child.type === "Bone") {
        boneMap.set(child.name, child as Bone);
      }
      if ((child as SkinnedMesh).isSkinnedMesh) {
        const skinned = child as SkinnedMesh;
        if (skinned.skeleton && skinned.skeleton.bones) {
          skinned.skeleton.bones.forEach((b) => {
            if (b.name) boneMap.set(b.name, b);
          });
        }
      }
    });

    const activeClip = animations.length > 0 ? animations[0] : null;
    let maxClipFrame = 120;

    if (activeClip) {
      maxClipFrame = Math.max(1, Math.ceil(activeClip.duration * fps));
    }

    const allSummaryFrames = new Set<number>();
    const boneTracks: BoneTrack[] = [];

    const boneNames = Array.from(boneMap.keys());

    if (boneNames.length === 0 && activeClip && activeClip.tracks.length > 0) {
      const nodeNames = new Set<string>();
      activeClip.tracks.forEach((track) => {
        const targetName = track.name.split(".")[0];
        if (targetName) nodeNames.add(targetName);
      });
      nodeNames.forEach((name) => boneNames.push(name));
    }

    boneNames.forEach((boneName, idx) => {
      const color = colors[idx % colors.length];
      const channels: BoneTrackChannel[] = [];
      const boneFrames = new Set<number>();

      if (activeClip) {
        activeClip.tracks.forEach((track) => {
          const parts = track.name.split(".");
          const trackTarget = parts[0];
          const trackProp = parts[1] || "property";

          if (trackTarget === boneName) {
            const channelKeyframes: KeyframePoint[] = [];
            for (let i = 0; i < track.times.length; i++) {
              const frame = Math.max(1, Math.round(track.times[i] * fps));
              boneFrames.add(frame);
              allSummaryFrames.add(frame);
              channelKeyframes.push({ frame });
            }

            let propColor = "#3b82f6";
            if (trackProp.includes("position")) propColor = "#ef4444";
            else if (trackProp.includes("quaternion") || trackProp.includes("rotation"))
              propColor = "#3b82f6";
            else if (trackProp.includes("scale")) propColor = "#22c55e";

            channels.push({
              id: `${boneName}_${trackProp}`,
              name: trackProp.charAt(0).toUpperCase() + trackProp.slice(1),
              color: propColor,
              keyframes: channelKeyframes.sort((a, b) => a.frame - b.frame),
            });
          }
        });
      }

      const sortedBoneFrames = Array.from(boneFrames)
        .sort((a, b) => a - b)
        .map((f) => ({ frame: f }));

      boneTracks.push({
        id: `bone_${boneName}`,
        name: boneName,
        type: "bone",
        color,
        expanded: false,
        keyframes: sortedBoneFrames,
        channels: channels.length > 0 ? channels : undefined,
      });
    });

    const sortedSummaryFrames = Array.from(allSummaryFrames)
      .sort((a, b) => a - b)
      .map((f) => ({ frame: f }));

    const tracks: BoneTrack[] = [
      {
        id: "summary",
        name: "Summary",
        type: "summary",
        color: "#f59e0b",
        expanded: true,
        keyframes: sortedSummaryFrames,
      },
      ...boneTracks,
    ];

    set({
      tracks,
      startFrame: 1,
      endFrame: maxClipFrame,
      currentFrame: 1,
    });
  },

  resetDefaultTracks: () =>
    set({
      tracks: DEFAULT_TRACKS,
      startFrame: 1,
      endFrame: 120,
      currentFrame: 1,
    }),
}));

export const useTimelineStore = useTimeStore;
