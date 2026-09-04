import { create } from "zustand";
import { Camera } from "three";
import { OrbitControls as OrbitControlsImpl } from "three-stdlib";

interface CameraStore {
  resetVersion: number;
  camera: Camera | null;
  controls: OrbitControlsImpl | null;

  requestReset: () => void;
  setCamera: (camera: Camera | null) => void;
  setControls: (controls: OrbitControlsImpl | null) => void;
}

export const useCameraStore = create<CameraStore>((set) => ({
  resetVersion: 0,
  camera: null,
  controls: null,

  requestReset: () =>
    set((state) => ({
      resetVersion: state.resetVersion + 1,
    })),
  setCamera: (camera) => set({ camera }),
  setControls: (controls) => set({ controls }),
}));