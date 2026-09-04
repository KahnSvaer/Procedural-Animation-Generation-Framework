import { useEffect, useRef } from "react";
import {
  OrbitControls,
  Grid,
  GizmoHelper,
  GizmoViewport,
} from "@react-three/drei";
import { useThree, useFrame } from "@react-three/fiber";
import { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { AnimationMixer } from "three";

import PlaceHolderCube from "./Cube";
import { useModelStore } from "../../stores/modelStore";
import { useCameraStore } from "../../stores/cameraStore";
import { useScreenshotStore } from "../../stores/screenshotStore";
import { useTimeStore } from "../../stores/timeStore";
import {
  DEFAULT_CAMERA_POSITION,
  DEFAULT_CAMERA_TARGET,
} from "../../constants/camera";

export const Scene = () => {
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const mixerRef = useRef<AnimationMixer | null>(null);
  const lastCapturedVersion = useRef(0);
  const isCapturing = useRef(false);

  const { camera, gl, scene } = useThree();

  const model = useModelStore((state) => state.model);
  const animations = useModelStore((state) => state.animations);
  const activeClipIndex = useModelStore((state) => state.activeClipIndex);

  const resetVersion = useCameraStore((state) => state.resetVersion);
  const screenshotVersion = useScreenshotStore((state) => state.screenshotVersion);

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

  useFrame(() => {
    if (mixerRef.current && animations.length > 0) {
      const { currentFrame, fps } = useTimeStore.getState();
      const time = Math.max(0, (currentFrame - 1) / fps);
      mixerRef.current.setTime(time);
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
      <ambientLight intensity={0.8} />
      <directionalLight position={[5, 5, 5]} intensity={2} />

      <Grid
        cellSize={0.5}
        sectionSize={2}
        fadeDistance={30}
        fadeStrength={1}
        infiniteGrid
      />

      {model ? (
        <primitive object={model} />
      ) : (
        <PlaceHolderCube />
      )}

      <GizmoHelper alignment="top-right">
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