import { useEffect, useRef, useMemo } from "react";
import {
  OrbitControls,
  Grid,
  GizmoHelper,
  GizmoViewport,
} from "@react-three/drei";
import { useThree, useFrame } from "@react-three/fiber";
import { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import {
  AnimationMixer,
  Bone,
  SkinnedMesh,
  Mesh,
  Object3D,
  InstancedMesh,
  Box3,
  Vector3,
  Quaternion,
  Color,
  CylinderGeometry,
  BufferAttribute,
  DirectionalLight,
  MeshBasicMaterial,
} from "three";
import { DEFAULT_BONE_COLOR } from "../../constants/timelineTheme";
import PlaceHolderCube from "./Cube";
import { useModelStore } from "../../stores/modelStore";
import { useCameraStore } from "../../stores/cameraStore";
import { useScreenshotStore } from "../../stores/screenshotStore";
import { useTimeStore } from "../../stores/timeStore";
import { useViewportStore } from "../../stores/viewportStore";
import {
  DEFAULT_CAMERA_POSITION,
  DEFAULT_CAMERA_TARGET,
} from "../../constants/camera";

const SkeletonVisualizer = ({ model }: { model: Object3D }) => {
  const cylinderMeshRef = useRef<InstancedMesh | null>(null);
  const sphereMeshRef = useRef<InstancedMesh | null>(null);

  const p1 = useMemo(() => new Vector3(), []);
  const p2 = useMemo(() => new Vector3(), []);
  const dir = useMemo(() => new Vector3(), []);
  const mid = useMemo(() => new Vector3(), []);
  const quat = useMemo(() => new Quaternion(), []);
  const up = useMemo(() => new Vector3(0, 1, 0), []);
  const scale = useMemo(() => new Vector3(1, 1, 1), []);
  const dummy = useMemo(() => new Object3D(), []);

  const { allBones, bonePairs } = useMemo(() => {
    const boneSet = new Set<Bone>();
    model.traverse((child) => {
      if ((child as Bone).isBone || child.type === "Bone") {
        boneSet.add(child as Bone);
      }
      if ((child as SkinnedMesh).isSkinnedMesh) {
        const skinned = child as SkinnedMesh;
        if (skinned.skeleton && skinned.skeleton.bones) {
          skinned.skeleton.bones.forEach((b) => {
            if (b) boneSet.add(b);
          });
        }
      }
    });

    const bones = Array.from(boneSet);
    const pairs: [Bone, Bone][] = [];

    bones.forEach((bone) => {
      if (bone.parent && boneSet.has(bone.parent as Bone)) {
        pairs.push([bone.parent as Bone, bone]);
      }
    });

    return { allBones: bones, bonePairs: pairs };
  }, [model]);

  const { cylinderRadius, ballRadius } = useMemo(() => {
    const box = new Box3().setFromObject(model);
    const size = new Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    const baseR = maxDim > 0 ? Math.max(0.006, Math.min(0.03, maxDim * 0.009)) : 0.011;
    return {
      cylinderRadius: baseR,
      ballRadius: baseR * 2.0,
    };
  }, [model]);

  const cylinderGeo = useMemo(() => {
    const geo = new CylinderGeometry(cylinderRadius, cylinderRadius, 1, 14, 16);
    const pos = geo.attributes.position;
    const count = pos.count;
    const colors = new Float32Array(count * 3);

    const colorBase = new Color(DEFAULT_BONE_COLOR);
    const colorTip = new Color("#a855f7");
    const tempColor = new Color();

    for (let i = 0; i < count; i++) {
      const y = pos.getY(i);
      const t = Math.max(0, Math.min(1, y + 0.5));
      tempColor.lerpColors(colorBase, colorTip, t);
      colors[i * 3] = tempColor.r;
      colors[i * 3 + 1] = tempColor.g;
      colors[i * 3 + 2] = tempColor.b;
    }

    geo.setAttribute("color", new BufferAttribute(colors, 3));
    return geo;
  }, [cylinderRadius]);

  useFrame(() => {
    if (cylinderMeshRef.current && bonePairs.length > 0) {
      for (let i = 0; i < bonePairs.length; i++) {
        const [parent, child] = bonePairs[i];
        parent.getWorldPosition(p1);
        child.getWorldPosition(p2);
        dir.subVectors(p2, p1);
        const len = dir.length();

        if (len > 0.0001) {
          mid.addVectors(p1, p2).multiplyScalar(0.5);
          dir.normalize();
          quat.setFromUnitVectors(up, dir);
          scale.set(1, len, 1);
          dummy.position.copy(mid);
          dummy.quaternion.copy(quat);
          dummy.scale.copy(scale);
          dummy.updateMatrix();
          cylinderMeshRef.current.setMatrixAt(i, dummy.matrix);
        } else {
          dummy.scale.set(0, 0, 0);
          dummy.updateMatrix();
          cylinderMeshRef.current.setMatrixAt(i, dummy.matrix);
        }
      }
      cylinderMeshRef.current.instanceMatrix.needsUpdate = true;
    }

    if (sphereMeshRef.current && allBones.length > 0) {
      for (let i = 0; i < allBones.length; i++) {
        allBones[i].getWorldPosition(dummy.position);
        dummy.quaternion.identity();
        dummy.scale.set(1, 1, 1);
        dummy.updateMatrix();
        sphereMeshRef.current.setMatrixAt(i, dummy.matrix);
      }
      sphereMeshRef.current.instanceMatrix.needsUpdate = true;
    }
  });

  return (
    <group>
      {bonePairs.length > 0 && (
        <instancedMesh
          ref={cylinderMeshRef}
          geometry={cylinderGeo}
          args={[undefined, undefined, bonePairs.length]}
          renderOrder={998}
        >
          <meshBasicMaterial
            vertexColors
            depthTest={false}
            transparent={false}
          />
        </instancedMesh>
      )}

      {allBones.length > 0 && (
        <instancedMesh
          ref={sphereMeshRef}
          args={[undefined, undefined, allBones.length]}
          renderOrder={1000}
        >
          <sphereGeometry args={[ballRadius, 16, 16]} />
          <meshBasicMaterial
            color={DEFAULT_BONE_COLOR}
            depthTest={false}
            transparent={false}
          />
        </instancedMesh>
      )}
    </group>
  );
};

export const Scene = () => {
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const mixerRef = useRef<AnimationMixer | null>(null);
  const headlightRef = useRef<DirectionalLight>(null);
  const lastCapturedVersion = useRef(0);
  const isCapturing = useRef(false);

  const { camera, gl, scene } = useThree();

  const model = useModelStore((state) => state.model);
  const animations = useModelStore((state) => state.animations);
  const activeClipIndex = useModelStore((state) => state.activeClipIndex);

  const resetVersion = useCameraStore((state) => state.resetVersion);
  const screenshotVersion = useScreenshotStore((state) => state.screenshotVersion);

  const isWireframe = useViewportStore((state) => state.isWireframe);
  const lightingMode = useViewportStore((state) => state.lightingMode);
  const showSkeleton = useViewportStore((state) => state.showSkeleton);
  const showGrid = useViewportStore((state) => state.showGrid);
  const showTextures = useViewportStore((state) => state.showTextures);

  useEffect(() => {
    controlsRef.current?.target.copy(DEFAULT_CAMERA_TARGET);
    controlsRef.current?.update();
  }, []);

  useEffect(() => {
    camera.position.copy(DEFAULT_CAMERA_POSITION);
    controlsRef.current?.target.copy(DEFAULT_CAMERA_TARGET);
    camera.updateProjectionMatrix();
    controlsRef.current?.update();
  }, [camera, resetVersion]);

  useEffect(() => {
    if (!model || animations.length === 0) {
      if (mixerRef.current) {
        mixerRef.current.stopAllAction();
        mixerRef.current = null;
      }
      return;
    }

    const mixer = new AnimationMixer(model);
    const clip = animations[activeClipIndex] || animations[0];
    const action = mixer.clipAction(clip);
    action.play();

    mixerRef.current = mixer;

    return () => {
      mixer.stopAllAction();
      mixerRef.current = null;
    };
  }, [model, animations, activeClipIndex]);

  useEffect(() => {
    if (!model) return;
    model.traverse((child) => {
      if ((child as Mesh).isMesh) {
        const mesh = child as Mesh;
        if (!mesh.userData) mesh.userData = {};

        if (!mesh.userData.originalMaterial && mesh.material) {
          mesh.userData.originalMaterial = mesh.material;
        }

        if (isWireframe) {
          if (!mesh.userData.wireframeMaterial) {
            mesh.userData.wireframeMaterial = new MeshBasicMaterial({
              color: "#38bdf8",
              wireframe: true,
            });
          }
          mesh.material = mesh.userData.wireframeMaterial;
        } else {
          if (mesh.userData.originalMaterial) {
            mesh.material = mesh.userData.originalMaterial;
          }

          const materials = Array.isArray(mesh.material)
            ? mesh.material
            : [mesh.material];

          materials.forEach((mat) => {
            if (!mat) return;
            if (!mat.userData) mat.userData = {};

            if (mat.userData.originalMap === undefined) {
              mat.userData.originalMap = "map" in mat ? (mat as unknown as { map: unknown }).map : null;
            }
            if (mat.userData.originalColor === undefined && "color" in mat) {
              const col = (mat as unknown as { color: { clone: () => unknown } }).color;
              mat.userData.originalColor = col ? col.clone() : null;
            }

            if ("map" in mat) {
              (mat as unknown as { map: unknown }).map =
                showTextures ? mat.userData.originalMap : null;
            }

            if ("color" in mat && mat.userData.originalColor) {
              const colObj = (mat as unknown as { color: { copy: (c: unknown) => void; set: (c: string) => void } }).color;
              if (showTextures) {
                colObj.copy(mat.userData.originalColor);
              } else {
                colObj.set("#d4d4d8");
              }
            }

            mat.needsUpdate = true;
          });
        }
      }
    });
  }, [model, isWireframe, showTextures]);

  useFrame(({ camera }) => {
    if (mixerRef.current && animations.length > 0) {
      const { currentFrame, fps } = useTimeStore.getState();
      const time = Math.max(0, (currentFrame - 1) / fps);
      mixerRef.current.setTime(time);
    }

    if (headlightRef.current) {
      headlightRef.current.position.copy(camera.position);
    }
  });

  useEffect(() => {
    if (
      screenshotVersion === 0 ||
      screenshotVersion === lastCapturedVersion.current ||
      isCapturing.current
    ) {
      return;
    }

    lastCapturedVersion.current = screenshotVersion;
    isCapturing.current = true;

    try {
      gl.render(scene, camera);

      const targetModelName = useModelStore.getState().modelName;
      const targetFrame = useTimeStore.getState().currentFrame;

      const filename = targetModelName
        ? targetModelName.replace(/\.[^/.]+$/, "")
        : "animgen_viewport";
      const fullFilename = `${filename}_f${targetFrame}.png`;

      const dataURL = gl.domElement.toDataURL("image/png");
      const link = document.createElement("a");
      link.href = dataURL;
      link.download = fullFilename;
      link.style.display = "none";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error("Screenshot capture error:", error);
    } finally {
      setTimeout(() => {
        isCapturing.current = false;
      }, 500);
    }
  }, [screenshotVersion, gl, scene, camera]);

  return (
    <>
      <directionalLight
        ref={headlightRef}
        intensity={lightingMode === "ambient" ? 1.35 : 0}
      />

      <directionalLight
        position={[8, 8, 8]}
        intensity={lightingMode === "studio" ? 2.5 : 0.9}
      />
      <directionalLight
        position={[-8, 4, 6]}
        intensity={lightingMode === "studio" ? 0.8 : 0.9}
      />
      <directionalLight
        position={[0, 8, -8]}
        intensity={lightingMode === "studio" ? 0.5 : 0.9}
      />
      <directionalLight
        position={[-8, -6, -8]}
        intensity={lightingMode === "studio" ? 0 : 0.9}
      />
      <directionalLight
        position={[8, -6, -8]}
        intensity={lightingMode === "studio" ? 0 : 0.9}
      />
      <directionalLight
        position={[0, -10, 0]}
        intensity={lightingMode === "studio" ? 0 : 0.9}
      />
      <directionalLight
        position={[0, 10, 0]}
        intensity={lightingMode === "studio" ? 0.2 : 0.9}
      />

      <ambientLight
        intensity={lightingMode === "studio" ? 0.35 : 1.8}
      />
      <hemisphereLight
        args={["#ffffff", "#64748b", lightingMode === "studio" ? 0.2 : 1.35]}
      />

      {showGrid && (
        <Grid
          cellSize={0.5}
          sectionSize={2}
          fadeDistance={30}
          fadeStrength={1}
          infiniteGrid
        />
      )}

      {model ? (
        <primitive object={model} />
      ) : (
        <PlaceHolderCube />
      )}

      {model && showSkeleton && <SkeletonVisualizer model={model} />}

      <GizmoHelper alignment="top-right" margin={[60, 60]}>
        <GizmoViewport />
      </GizmoHelper>

      <OrbitControls
        ref={controlsRef}
        makeDefault
        enableDamping
        dampingFactor={0.1}
      />
    </>
  );
};

export default Scene;