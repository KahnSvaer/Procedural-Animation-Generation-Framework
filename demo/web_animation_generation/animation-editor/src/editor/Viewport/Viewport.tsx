import { Canvas } from "@react-three/fiber";
import Scene from "./Scene";

function Viewport() {
  return (
    <section className="viewport">
      <Canvas
        style={{
          width: "100%",
          height: "100%",
        }}
        camera={{
          position: [0, 0, 5],
          fov: 50,
        }}
      >
        <Scene />
      </Canvas>
    </section>
  );
}

export default Viewport;