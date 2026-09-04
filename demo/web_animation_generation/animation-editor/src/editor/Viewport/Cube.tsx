import { useViewportStore } from "../../stores/viewportStore";

function PlaceHolderCube() {
  const isWireframe = useViewportStore((state) => state.isWireframe);
  const showTextures = useViewportStore((state) => state.showTextures);

  return (
    <mesh>
      <boxGeometry args={[1, 1, 1]} />
      {isWireframe ? (
        <meshBasicMaterial color="#38bdf8" wireframe />
      ) : (
        <meshStandardMaterial
          color={showTextures ? "orange" : "#d4d4d8"}
        />
      )}
    </mesh>
  );
}

export default PlaceHolderCube;