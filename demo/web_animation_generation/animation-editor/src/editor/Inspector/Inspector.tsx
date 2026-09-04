import { Layers, Bone as BoneIcon, Info } from "lucide-react";
import "./Inspector.css";
import { useModelStore } from "../../stores/modelStore";
import { useTimeStore } from "../../stores/timeStore";

export const Inspector = () => {
  const { modelName, animations, stats } = useModelStore();
  const { currentFrame, selectedBoneId, tracks } = useTimeStore();

  const activeTrack = tracks.find((t) => t.id === selectedBoneId);
  const boneTracks = tracks.filter((t) => t.type === "bone");

  return (
    <aside className="inspector">
      <div className="inspector-header">
        <h2>Inspector</h2>
        {modelName && <span className="toolbar-badge">{modelName}</span>}
      </div>

      <div className="inspector-content">
        <section className="inspector-section">
          <div className="section-title">
            <Info className="w-3.5 h-3.5 text-sky-400" />
            <span>Model Overview</span>
          </div>
          <div className="section-body">
            <div className="stat-row">
              <span>Name</span>
              <span className="stat-value">{modelName || "Default Cube"}</span>
            </div>
            <div className="stat-row">
              <span>Vertices</span>
              <span className="stat-value">
                {stats ? stats.vertices.toLocaleString() : "8"}
              </span>
            </div>
            <div className="stat-row">
              <span>Triangles / Faces</span>
              <span className="stat-value">
                {stats ? stats.triangles.toLocaleString() : "12"}
              </span>
            </div>
            <div className="stat-row">
              <span>Armature Bones</span>
              <span className="stat-value">{boneTracks.length}</span>
            </div>
            <div className="stat-row">
              <span>Animation Clips</span>
              <span className="stat-value">{animations.length}</span>
            </div>
            <div className="stat-row">
              <span>Dimensions (X, Y, Z)</span>
              <span className="stat-value">
                {stats
                  ? `${stats.size.x} × ${stats.size.y} × ${stats.size.z}`
                  : "1.00 × 1.00 × 1.00"}
              </span>
            </div>
          </div>
        </section>

        <section className="inspector-section">
          <div className="section-title">
            <Layers className="w-3.5 h-3.5 text-amber-400" />
            <span>Active Selection</span>
          </div>
          <div className="section-body">
            <div className="stat-row">
              <span>Current Frame</span>
              <span className="stat-value text-sky-400">{currentFrame}</span>
            </div>
            <div className="stat-row">
              <span>Active Track</span>
              <span className="stat-value">
                {activeTrack ? activeTrack.name : "Summary"}
              </span>
            </div>
            <div className="stat-row">
              <span>Keyframes</span>
              <span className="stat-value">
                {activeTrack ? activeTrack.keyframes.length : 0}
              </span>
            </div>
          </div>
        </section>

        {boneTracks.length > 0 && (
          <section className="inspector-section">
            <div className="section-title">
              <BoneIcon className="w-3.5 h-3.5 text-emerald-400" />
              <span>Armature Bones ({boneTracks.length})</span>
            </div>
            <div className="section-body" style={{ maxHeight: "240px", overflowY: "auto" }}>
              {boneTracks.map((b) => (
                <div key={b.id} className="stat-row">
                  <span>{b.name}</span>
                  <span className="stat-value">{b.keyframes.length} keys</span>
                </div>
              ))}
            </div>
          </section>
        )}
      </div>
    </aside>
  );
};

export default Inspector;