import { useCameraStore } from "../../stores/cameraStore";
import { useModelStore } from "../../stores/modelStore";
import { useScreenshotStore } from "../../stores/screenshotStore";

import { useRef } from "react";

import FilePicker from "../Import/FilePicker";
import { importGLTFFromFile } from "../Import/GLTFImporter";

function Toolbar() {
  const filePickerRef = useRef<HTMLInputElement>(null);

  const requestReset = useCameraStore(
    (state) => state.requestReset
  );

  const { setModel } = useModelStore();

  const handleFileSelected = async (file: File) => {
    try {
        const model = await importGLTFFromFile(file);

        setModel(model, file.name);
    } catch (error) {
        console.error(error);
    }
  };

  const requestScreenshot = useScreenshotStore(
    (state) => state.requestScreenshot
  );

  return (
    <header className="toolbar">
      <button onClick={() => filePickerRef.current?.click()}>
          Open
      </button>

      <button onClick={requestReset}>
        Reset Camera
      </button>

      <button onClick={requestScreenshot}>
          {/* Later change it to export with screenshot just being one of the options inside the dropdown */}
          Screenshot 
      </button>

      <FilePicker ref={filePickerRef} onFileSelected={(file) => handleFileSelected(file)} />

    </header>

  
  );
}

export default Toolbar;