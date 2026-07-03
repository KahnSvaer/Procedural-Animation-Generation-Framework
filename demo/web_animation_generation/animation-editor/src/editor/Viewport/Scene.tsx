import {
    OrbitControls,
    Grid,
    GizmoHelper,
    GizmoViewport
} from "@react-three/drei";
import Cube from "./Cube";

import { useEffect, useRef } from "react";
import { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { useCameraStore } from "../../stores/cameraStore";
import {
    DEFAULT_CAMERA_POSITION,
    DEFAULT_CAMERA_TARGET,
} from "../../constants/camera";
import { useThree } from "@react-three/fiber";

function Scene() {
    const controlsRef = useRef<OrbitControlsImpl>(null);
    const { camera } = useThree();

    const resetVersion = useCameraStore(
        (state) => state.resetVersion
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

            <Cube />

            <GizmoHelper alignment="top-right">
                <GizmoViewport />
            </GizmoHelper>

            <OrbitControls ref={controlsRef} />
        </>
    );
}

export default Scene;