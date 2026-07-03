import { GLTFLoader, type GLTF } from "three/examples/jsm/loaders/GLTFLoader.js";
import { Group } from "three";

const loader = new GLTFLoader();

export function importGLTF(url: string): Promise<Group> {
    return new Promise((resolve, reject) => {
        loader.load(
            url,
            (gltf: GLTF) => {
                resolve(gltf.scene);
            },
            undefined,
            (error: unknown) => {
                reject(error);
            }
        );
    });
}

export async function importGLTFFromFile(file: File): Promise<Group> {
    const url = URL.createObjectURL(file);

    try {
        return await importGLTF(url);
    } finally {
        URL.revokeObjectURL(url);
    }
}
