import { GLTFLoader, type GLTF } from "three/examples/jsm/loaders/GLTFLoader.js";
import { Group } from "three";

const loader = new GLTFLoader();

export function importGLTF(file: File): Promise<Group> {
    return new Promise((resolve, reject) => {
        const url = URL.createObjectURL(file);

        loader.load(
            url,
            (gltf: GLTF) => {
                URL.revokeObjectURL(url);
                resolve(gltf.scene);
            },
            undefined,
            (error: unknown) => {
                URL.revokeObjectURL(url);
                reject(error);
            }
        );
    });
}