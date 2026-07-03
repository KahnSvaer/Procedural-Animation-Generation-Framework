import {
    OrbitControls,
    Grid,
    GizmoHelper,
    GizmoViewport
} from "@react-three/drei";

import Cube from "./Cube";

function Scene() {
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

            <OrbitControls />
        </>
    );
}

export default Scene;