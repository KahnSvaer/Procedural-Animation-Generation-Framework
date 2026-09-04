import { create } from "zustand";
import { Group, Object3D, Box3, Vector3, Mesh, AnimationClip } from "three";
import { useTimeStore } from "./timeStore";

export interface ModelStats {
  vertices: number;
  triangles: number;
  size: { x: string; y: string; z: string };
}

interface ModelStore {
  model: Object3D | null;
  modelName: string | null;
  animations: AnimationClip[];
  activeClipIndex: number;
  stats: ModelStats | null;

  setModel: (model: Group, animations?: AnimationClip[], modelName?: string) => void;
  setActiveClipIndex: (index: number) => void;
  clearModel: () => void;
}

export const useModelStore = create<ModelStore>()((set, get) => ({
  model: null,
  modelName: null,
  animations: [],
  activeClipIndex: 0,
  stats: null,

  setModel: (model, animations = [], modelName) => {
    let vertices = 0;
    let triangles = 0;

    model.traverse((child) => {
      if ((child as Mesh).isMesh) {
        const mesh = child as Mesh;
        if (mesh.geometry) {
          const pos = mesh.geometry.attributes.position;
          if (pos) vertices += pos.count;
          if (mesh.geometry.index) {
            triangles += mesh.geometry.index.count / 3;
          } else if (pos) {
            triangles += pos.count / 3;
          }
        }
      }
    });

    const box = new Box3().setFromObject(model);
    const size = new Vector3();
    box.getSize(size);

    const stats: ModelStats = {
      vertices,
      triangles: Math.round(triangles),
      size: {
        x: size.x.toFixed(2),
        y: size.y.toFixed(2),
        z: size.z.toFixed(2),
      },
    };

    useTimeStore.getState().populateFromGLTF(model, animations);

    set({
      model,
      modelName: modelName ?? "Model",
      animations,
      activeClipIndex: 0,
      stats,
    });
  },

  setActiveClipIndex: (index: number) => {
    const { model, animations } = get();
    if (model && animations.length > index) {
      useTimeStore.getState().populateFromGLTF(model, [animations[index]]);
      set({ activeClipIndex: index });
    }
  },

  clearModel: () => {
    useTimeStore.getState().resetDefaultTracks();
    set({
      model: null,
      modelName: null,
      animations: [],
      activeClipIndex: 0,
      stats: null,
    });
  },
}));