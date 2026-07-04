import { create } from "zustand";
import { Group, Object3D } from "three";

interface ModelStore {
    model: Object3D | null;
    modelName: string | null;

    setModel: (model: Group, modelName?: string) => void;
    clearModel: () => void;
}

export const useModelStore = create<ModelStore>()((set) => ({
    model: null,
    modelName: null,

    setModel: (model, modelName) =>
    set({
        model,
        modelName: modelName ?? "Model",
    }),

    clearModel: () => set({ model: null, modelName: null }),
}));