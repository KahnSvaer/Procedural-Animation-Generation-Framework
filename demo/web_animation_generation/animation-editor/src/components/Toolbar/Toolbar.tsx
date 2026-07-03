import { useCameraStore } from "../../stores/cameraStore";


function Toolbar() {

  const requestReset = useCameraStore(
    (state) => state.requestReset
  );

  return (
    <header className="toolbar">
      <button>Open</button>
      <button onClick={requestReset}>Reset Camera</button>
      <button>Export</button>
    </header>
  );
}

export default Toolbar;