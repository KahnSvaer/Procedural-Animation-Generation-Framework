import { create } from "zustand";

interface CameraStore {
    resetVersion: number;

    requestReset: () => void;
}

export const useCameraStore = create<CameraStore>((set) => ({
    resetVersion: 0,

    requestReset: () =>
        set((state) => ({
            resetVersion: state.resetVersion + 1,
        })),
}));