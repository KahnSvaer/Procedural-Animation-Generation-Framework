import { GLTFLoader, type GLTF } from "three/examples/jsm/loaders/GLTFLoader.js";
import { Group, AnimationClip } from "three";

export interface ImportedGLTFData {
  scene: Group;
  animations: AnimationClip[];
}

const loader = new GLTFLoader();

export function importGLTF(url: string): Promise<ImportedGLTFData> {
  return new Promise((resolve, reject) => {
    loader.load(
      url,
      (gltf: GLTF) => {
        resolve({
          scene: gltf.scene,
          animations: gltf.animations || [],
        });
      },
      undefined,
      (error: unknown) => {
        reject(error);
      }
    );
  });
}

export async function importGLTFFromFile(file: File): Promise<ImportedGLTFData> {
  const url = URL.createObjectURL(file);

  try {
    return await importGLTF(url);
  } finally {
    URL.revokeObjectURL(url);
  }
}
