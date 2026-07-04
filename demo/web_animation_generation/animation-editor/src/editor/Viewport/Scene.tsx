import {
    OrbitControls,
    Grid,
    GizmoHelper,
    GizmoViewport
} from "@react-three/drei";
import PlaceHolderCube from "./Cube";
import { useModelStore } from "../../stores/modelStore";

import { useEffect, useRef } from "react";
import { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { useCameraStore } from "../../stores/cameraStore";
import {
    DEFAULT_CAMERA_POSITION,
    DEFAULT_CAMERA_TARGET,
} from "../../constants/camera";
import { useThree } from "@react-three/fiber";

import { useScreenshotStore } from "../../stores/screenshotStore";

function Scene() {
    const controlsRef = useRef<OrbitControlsImpl>(null);
    const { camera, gl } = useThree();

    const model = useModelStore((state) => state.model);

    const resetVersion = useCameraStore(
        (state) => state.resetVersion
    );

    const screenshotVersion = useScreenshotStore(
        (state) => state.screenshotVersion
    );

    const modelName = useModelStore(
        (state) => state.modelName
    );


    useEffect(() => {
        controlsRef.current?.target.copy(DEFAULT_CAMERA_TARGET);
        controlsRef.current?.update();
    }, []);

    useEffect(() => {
        camera.position.copy(DEFAULT_CAMERA_POSITION);
        controlsRef.current?.target.copy(DEFAULT_CAMERA_TARGET);
        camera.updateProjectionMatrix();
        controlsRef.current?.update();
        console.log("Camera reset requested.");
    }, [camera, resetVersion]);

    useEffect(() => {
        if (screenshotVersion === 0) return;
        const dataURL = gl.domElement.toDataURL("image/png");
        const link = document.createElement("a");
        const filename = modelName
            ? modelName.replace(/\.[^/.]+$/, "")
            : "Untitled Model";
        link.download = `${filename}.png`;
        link.href = dataURL;
        link.click();
    }, [screenshotVersion]);

    return (
        <>
            <ambientLight intensity={0.8} />

            <directionalLight
                position={[5, 5, 5]}
                intensity={2}
            />

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

            <OrbitControls ref={controlsRef} />
        </>
    );
}

export default Scene;