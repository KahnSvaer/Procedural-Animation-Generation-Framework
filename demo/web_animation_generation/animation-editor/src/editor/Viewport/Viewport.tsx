import { Canvas } from "@react-three/fiber";
import "./Viewport.css";
import Scene from "./Scene";
import ViewportOverlay from "./ViewportOverlay";
import Gizmo from "./Gizmo";

function Viewport() {
  return (
    <section className="viewport">
      <Gizmo />
      <ViewportOverlay />
      <Canvas
        gl={{
          preserveDrawingBuffer: true,
          powerPreference: "high-performance",
          alpha: true,
        }}
        dpr={[1, 1.5]}
        style={{
          width: "100%",
          height: "100%",
        }}
      >
        <Scene />
      </Canvas>
    </section>
  );
}

export default Viewport;