function PlaceHolderCube() {
  return (
    <mesh>
      <boxGeometry args={[1, 1, 1]} />

      <meshStandardMaterial color="orange" />
    </mesh>
  );
}

export default PlaceHolderCube;