import React, { useRef } from "react";
import "./Toolbar.css";
import {
  FolderOpen,
  Camera,
  RotateCcw,
} from "lucide-react";
import { useCameraStore } from "../../stores/cameraStore";
import { useModelStore } from "../../stores/modelStore";
import { useScreenshotStore } from "../../stores/screenshotStore";
import FilePicker from "../Import/FilePicker";
import { importGLTFFromFile } from "../Import/GLTFImporter";

export const Toolbar: React.FC = () => {
  const filePickerRef = useRef<HTMLInputElement>(null);

  const requestReset = useCameraStore((state) => state.requestReset);
  const { setModel } = useModelStore();
  const requestScreenshot = useScreenshotStore((state) => state.requestScreenshot);

  const handleFileSelected = async (file: File) => {
    try {
      const { scene, animations } = await importGLTFFromFile(file);
      setModel(scene, animations, file.name);
    } catch (error) {
      console.error("Error loading GLTF:", error);
    }
  };

  return (
    <header className="toolbar">
      <div className="toolbar-brand">
        <span>AnimGen Studio</span>
      </div>

      <div className="toolbar-actions">
        <button
          className="toolbar-btn"
          onClick={() => filePickerRef.current?.click()}
          title="Open GLB/GLTF 3D Model"
        >
          <FolderOpen className="w-3.5 h-3.5" />
          <span>Open File</span>
        </button>

        <button className="toolbar-btn" onClick={requestReset} title="Reset 3D Camera View">
          <RotateCcw className="w-3.5 h-3.5" />
          <span>Reset Camera</span>
        </button>

        <button
          className="toolbar-btn primary"
          onClick={requestScreenshot}
          title="Capture High-Res Viewport Screenshot"
        >
          <Camera className="w-3.5 h-3.5" />
          <span>Screenshot</span>
        </button>

        <FilePicker
          ref={filePickerRef}
          onFileSelected={(file) => handleFileSelected(file)}
        />
      </div>
    </header>
  );
};

export default Toolbar;