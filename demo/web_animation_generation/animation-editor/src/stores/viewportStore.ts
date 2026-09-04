import { create } from "zustand";

export type LightingMode = "studio" | "ambient";

interface ViewportStore {
  isWireframe: boolean;
  lightingMode: LightingMode;
  showSkeleton: boolean;
  showGrid: boolean;
  showTextures: boolean;

  toggleWireframe: () => void;
  setWireframe: (value: boolean) => void;
  toggleLightingMode: () => void;
  setLightingMode: (mode: LightingMode) => void;
  toggleSkeleton: () => void;
  setSkeleton: (value: boolean) => void;
  toggleGrid: () => void;
  setGrid: (value: boolean) => void;
  toggleTextures: () => void;
  setTextures: (value: boolean) => void;
}

export const useViewportStore = create<ViewportStore>((set) => ({
  isWireframe: false,
  lightingMode: "studio",
  showSkeleton: true,
  showGrid: true,
  showTextures: true,

  toggleWireframe: () => set((state) => ({ isWireframe: !state.isWireframe })),
  setWireframe: (isWireframe) => set({ isWireframe }),

  toggleLightingMode: () =>
    set((state) => ({
      lightingMode: state.lightingMode === "studio" ? "ambient" : "studio",
    })),
  setLightingMode: (lightingMode) => set({ lightingMode }),

  toggleSkeleton: () => set((state) => ({ showSkeleton: !state.showSkeleton })),
  setSkeleton: (showSkeleton) => set({ showSkeleton }),

  toggleGrid: () => set((state) => ({ showGrid: !state.showGrid })),
  setGrid: (showGrid) => set({ showGrid }),

  toggleTextures: () => set((state) => ({ showTextures: !state.showTextures })),
  setTextures: (showTextures) => set({ showTextures }),
}));
