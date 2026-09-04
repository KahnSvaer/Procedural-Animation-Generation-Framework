import { Canvas } from "@react-three/fiber";
import "./Viewport.css";
import Scene from "./Scene";

function Viewport() {
  return (
    <section className="viewport">
      <Canvas
        gl={{
          preserveDrawingBuffer: true,
          powerPreference: "high-performance",
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