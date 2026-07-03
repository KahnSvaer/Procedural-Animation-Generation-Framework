import { create } from "zustand";
import { Object3D } from "three";

interface ModelStore {
    model: Object3D | null;

    setModel: (model: Object3D) => void;
    clearModel: () => void;
}

export const useModelStore = create<ModelStore>()((set) => ({
    model: null,

    setModel: (model) => set({ model }),

    clearModel: () => set({ model: null }),
}));