import { useCameraStore } from "../../stores/cameraStore";
import { useModelStore } from "../../stores/modelStore";

import { useRef } from "react";

import FilePicker from "../Import/FilePicker";
import { importGLTF } from "../Import/GLTFImporter";

function Toolbar() {
  const filePickerRef = useRef<HTMLInputElement>(null);

  const requestReset = useCameraStore(
    (state) => state.requestReset
  );

  const { setModel } = useModelStore();

  const handleFileSelected = async (file: File) => {
    try {
        const model = await importGLTF(file);

        setModel(model);
    } catch (error) {
        console.error(error);
    }
};

  return (
    <header className="toolbar">
      <button onClick={() => filePickerRef.current?.click()}>
          Open
      </button>

      <button onClick={requestReset}>
        Reset Camera
      </button>

      <button>Export</button>

      <FilePicker ref={filePickerRef} onFileSelected={(file) => handleFileSelected(file)} />

    </header>

  
  );
}

export default Toolbar;