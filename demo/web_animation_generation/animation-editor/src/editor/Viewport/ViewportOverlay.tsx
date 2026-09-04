import React from "react";
import {
  Box,
  Sun,
  SunDim,
  Bone,
  Grid3X3,
  Palette,
} from "lucide-react";
import { useViewportStore } from "../../stores/viewportStore";

export const ViewportOverlay: React.FC = () => {
  const {
    isWireframe,
    lightingMode,
    showSkeleton,
    showGrid,
    showTextures,
    toggleWireframe,
    toggleLightingMode,
    toggleSkeleton,
    toggleGrid,
    toggleTextures,
  } = useViewportStore();

  return (
    <div className="viewport-overlay-toolbar">
      <button
        className={`viewport-tool-btn ${isWireframe ? "active" : ""}`}
        onClick={toggleWireframe}
        title={isWireframe ? "Shading: Wireframe (Click for Solid)" : "Shading: Solid (Click for Wireframe)"}
      >
        <Box className="w-3.5 h-3.5" />
      </button>

      <button
        className={`viewport-tool-btn ${showTextures ? "active" : ""}`}
        onClick={toggleTextures}
        title={showTextures ? "Textures: Enabled (Click for Clay/Untextured)" : "Textures: Disabled / Clay (Click for Textured)"}
      >
        <Palette className="w-3.5 h-3.5" />
      </button>

      <button
        className={`viewport-tool-btn ${lightingMode === "ambient" ? "active" : ""}`}
        onClick={toggleLightingMode}
        title={
          lightingMode === "studio"
            ? "Lighting: Studio (Click for Ambient)"
            : "Lighting: Ambient (Click for Studio)"
        }
      >
        {lightingMode === "studio" ? (
          <Sun className="w-3.5 h-3.5" />
        ) : (
          <SunDim className="w-3.5 h-3.5" />
        )}
      </button>

      <button
        className={`viewport-tool-btn ${showSkeleton ? "active" : ""}`}
        onClick={toggleSkeleton}
        title={showSkeleton ? "Hide Skeleton Armature" : "Show Skeleton Armature"}
      >
        <Bone className="w-3.5 h-3.5" />
      </button>

      <button
        className={`viewport-tool-btn ${showGrid ? "active" : ""}`}
        onClick={toggleGrid}
        title={showGrid ? "Hide Ground Grid" : "Show Ground Grid"}
      >
        <Grid3X3 className="w-3.5 h-3.5" />
      </button>
    </div>
  );
};

export default ViewportOverlay;
