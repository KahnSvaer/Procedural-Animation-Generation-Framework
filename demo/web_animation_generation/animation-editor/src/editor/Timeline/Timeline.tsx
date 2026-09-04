import React, { useRef, useEffect, useCallback, useMemo } from "react";
import "./Timeline.css";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  ChevronLeft,
  ChevronRight,
  Repeat,
  ZoomIn,
  ZoomOut,
  Plus,
  Trash2,
  Eye,
  EyeOff,
  Lock,
  Unlock,
  ChevronDown,
  ChevronRight as ChevronRightIcon,
  Bone as BoneIcon,
  Layers,
  Sliders,
  Film,
} from "lucide-react";
import { useTimeStore } from "../../stores/timeStore";
import { useModelStore } from "../../stores/modelStore";
import {
  FPS_OPTIONS,
  PLAYBACK_SPEED_OPTIONS,
  SUMMARY_TRACK_COLOR,
  DEFAULT_BONE_COLOR,
  SELECTED_KEYFRAME_COLOR,
} from "../../constants/timelineTheme";

export const Timeline: React.FC = () => {
  const {
    currentFrame,
    startFrame,
    endFrame,
    fps,
    isPlaying,
    isLooping,
    playbackSpeed,
    zoom,
    tracks,
    selectedBoneId,
    selectedKeyframe,
    setCurrentFrame,
    setStartFrame,
    setEndFrame,
    setFps,
    togglePlay,
    setIsLooping,
    setPlaybackSpeed,
    stepForward,
    stepBackward,
    goToStart,
    goToEnd,
    setZoom,
    setSelectedBoneId,
    setSelectedKeyframe,
    toggleTrackExpanded,
    toggleTrackMuted,
    toggleTrackLocked,
    addKeyframe,
    removeKeyframe,
  } = useTimeStore();

  const { animations, activeClipIndex, setActiveClipIndex } = useModelStore();

  const rulerRef = useRef<HTMLDivElement>(null);
  const gridContainerRef = useRef<HTMLDivElement>(null);
  const isDraggingPlayhead = useRef(false);

  // Playback timer loop
  useEffect(() => {
    if (!isPlaying) return;

    const interval = 1000 / (fps * playbackSpeed);
    const timer = setInterval(() => {
      stepForward(1);
    }, interval);

    return () => clearInterval(timer);
  }, [isPlaying, fps, playbackSpeed, stepForward]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes((e.target as HTMLElement)?.tagName)) {
        return;
      }

      if (e.code === "Space") {
        e.preventDefault();
        togglePlay();
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        if (e.shiftKey) goToStart();
        else stepBackward(1);
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        if (e.shiftKey) goToEnd();
        else stepForward(1);
      } else if (e.key === "i" || e.key === "I") {
        e.preventDefault();
        if (selectedBoneId) {
          addKeyframe(selectedBoneId, currentFrame);
        } else {
          addKeyframe("summary", currentFrame);
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [togglePlay, stepBackward, stepForward, goToStart, goToEnd, selectedBoneId, currentFrame, addKeyframe]);

  const getFrameFromMouseX = useCallback(
    (clientX: number): number => {
      if (!gridContainerRef.current) return startFrame;
      const rect = gridContainerRef.current.getBoundingClientRect();
      const scrollLeft = gridContainerRef.current.scrollLeft;
      const x = clientX - rect.left + scrollLeft;
      const frame = Math.round(x / zoom) + startFrame;
      return Math.max(startFrame, Math.min(endFrame, frame));
    },
    [zoom, startFrame, endFrame]
  );

  const handleRulerMouseDown = (e: React.MouseEvent) => {
    isDraggingPlayhead.current = true;
    const frame = getFrameFromMouseX(e.clientX);
    setCurrentFrame(frame);

    const handleMouseMove = (moveEvent: MouseEvent) => {
      if (!isDraggingPlayhead.current) return;
      const f = getFrameFromMouseX(moveEvent.clientX);
      setCurrentFrame(f);
    };

    const handleMouseUp = () => {
      isDraggingPlayhead.current = false;
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
  };

  const totalFrames = endFrame - startFrame + 1;
  const contentWidth = useMemo(() => Math.max(800, totalFrames * zoom + 120), [totalFrames, zoom]);

  const ticks = useMemo(() => {
    const tickInterval = zoom < 8 ? 20 : zoom < 16 ? 10 : zoom < 30 ? 5 : 1;
    const list: { frame: number; isMajor: boolean }[] = [];
    for (let f = startFrame; f <= endFrame; f++) {
      if (f % tickInterval === 0 || f === startFrame || f === endFrame) {
        list.push({ frame: f, isMajor: f % (tickInterval * 2) === 0 || f === startFrame || f === endFrame });
      }
    }
    return list;
  }, [startFrame, endFrame, zoom]);

  const playheadPositionX = (currentFrame - startFrame) * zoom;

  return (
    <div className="blender-timeline">
      <div className="timeline-header">
        <div className="timeline-header-left">
          {animations.length > 0 && (
            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <Film className="w-3.5 h-3.5 text-orange-400" />
              <select
                value={activeClipIndex}
                onChange={(e) => setActiveClipIndex(Number(e.target.value))}
                className="timeline-select"
                title="Active Animation Clip"
              >
                {animations.map((clip, idx) => (
                  <option key={idx} value={idx}>
                    {clip.name || `Clip ${idx + 1}`} ({clip.duration.toFixed(2)}s)
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="timeline-fps-badge">
            <select
              value={fps}
              onChange={(e) => setFps(Number(e.target.value))}
              className="timeline-select"
              title="Frames Per Second"
            >
              {FPS_OPTIONS.map((fpsVal) => (
                <option key={fpsVal} value={fpsVal}>
                  {fpsVal} FPS
                </option>
              ))}
            </select>
          </div>

          <div className="timeline-speed-badge">
            <select
              value={playbackSpeed}
              onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
              className="timeline-select"
              title="Playback Speed"
            >
              {PLAYBACK_SPEED_OPTIONS.map((speedVal) => (
                <option key={speedVal} value={speedVal}>
                  {speedVal}x
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="timeline-transport">
          <button
            className="timeline-btn"
            onClick={goToStart}
            title="Jump to Start (Shift+Left)"
          >
            <SkipBack className="w-3.5 h-3.5" />
          </button>

          <button
            className="timeline-btn"
            onClick={() => stepBackward(1)}
            title="Previous Frame (Left Arrow)"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>

          <button
            className={`timeline-btn timeline-play-btn ${isPlaying ? "active" : ""}`}
            onClick={togglePlay}
            title={isPlaying ? "Pause (Space)" : "Play (Space)"}
          >
            {isPlaying ? (
              <Pause className="w-4 h-4 fill-current" />
            ) : (
              <Play className="w-4 h-4 fill-current ml-0.5" />
            )}
          </button>

          <button
            className="timeline-btn"
            onClick={() => stepForward(1)}
            title="Next Frame (Right Arrow)"
          >
            <ChevronRight className="w-4 h-4" />
          </button>

          <button
            className="timeline-btn"
            onClick={goToEnd}
            title="Jump to End (Shift+Right)"
          >
            <SkipForward className="w-3.5 h-3.5" />
          </button>

          <button
            className={`timeline-btn ${isLooping ? "active-loop" : ""}`}
            onClick={() => setIsLooping(!isLooping)}
            title="Toggle Loop"
          >
            <Repeat className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="timeline-header-right">
          <div className="frame-input-group">
            <span className="frame-label">Frame</span>
            <input
              type="number"
              className="frame-input current-frame-input"
              value={currentFrame}
              onChange={(e) => setCurrentFrame(Number(e.target.value))}
              min={startFrame}
              max={endFrame}
            />
          </div>

          <div className="frame-range-group">
            <span className="frame-label text-muted">Range</span>
            <input
              type="number"
              className="frame-input range-input"
              value={startFrame}
              onChange={(e) => setStartFrame(Number(e.target.value))}
            />
            <span className="range-separator">-</span>
            <input
              type="number"
              className="frame-input range-input"
              value={endFrame}
              onChange={(e) => setEndFrame(Number(e.target.value))}
            />
          </div>

          <div className="timeline-actions">
            <button
              className="timeline-btn-action"
              onClick={() => {
                const targetTrack = selectedBoneId || "summary";
                addKeyframe(targetTrack, currentFrame);
              }}
              title="Insert Keyframe (Key 'I')"
            >
              <Plus className="w-3.5 h-3.5 mr-1" />
              <span>Key</span>
            </button>

            {selectedKeyframe && (
              <button
                className="timeline-btn-action danger"
                onClick={() => {
                  removeKeyframe(selectedKeyframe.trackId, selectedKeyframe.frame);
                  setSelectedKeyframe(null);
                }}
                title="Delete Selected Keyframe"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            )}

            <div className="timeline-zoom-controls">
              <button
                className="timeline-btn-small"
                onClick={() => setZoom(zoom - 2)}
                title="Zoom Out"
              >
                <ZoomOut className="w-3 h-3" />
              </button>
              <span className="zoom-value">{Math.round(zoom)}px</span>
              <button
                className="timeline-btn-small"
                onClick={() => setZoom(zoom + 2)}
                title="Zoom In"
              >
                <ZoomIn className="w-3 h-3" />
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="timeline-body">
        <div className="timeline-sidebar">
          <div className="timeline-sidebar-header">
            <span>Channels</span>
            <span className="track-count">{tracks.length}</span>
          </div>

          <div className="timeline-track-list">
            {tracks.map((track) => (
              <React.Fragment key={track.id}>
                <div
                  className={`track-item ${track.type} ${
                    selectedBoneId === track.id ? "selected" : ""
                  }`}
                  onClick={() => setSelectedBoneId(track.id)}
                >
                  <div className="track-toggle">
                    {track.channels && track.channels.length > 0 ? (
                      <button
                        className="track-expand-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleTrackExpanded(track.id);
                        }}
                      >
                        {track.expanded ? (
                          <ChevronDown className="w-3 h-3" />
                        ) : (
                          <ChevronRightIcon className="w-3 h-3" />
                        )}
                      </button>
                    ) : (
                      <span className="track-bullet" style={{ background: track.color }} />
                    )}
                  </div>

                  <div className="track-icon-wrapper">
                    {track.type === "summary" && <Layers className="w-3.5 h-3.5 text-amber-400" />}
                    {track.type === "bone" && <BoneIcon className="w-3.5 h-3.5 text-sky-400" />}
                    {track.type === "property" && <Sliders className="w-3.5 h-3.5 text-emerald-400" />}
                  </div>

                  <span className="track-name" title={track.name}>
                    {track.name}
                  </span>

                  <div className="track-controls" onClick={(e) => e.stopPropagation()}>
                    <button
                      className={`track-ctrl-btn ${track.muted ? "muted" : ""}`}
                      onClick={() => toggleTrackMuted(track.id)}
                      title={track.muted ? "Unmute Track" : "Mute Track"}
                    >
                      {track.muted ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
                    </button>
                    <button
                      className={`track-ctrl-btn ${track.locked ? "locked" : ""}`}
                      onClick={() => toggleTrackLocked(track.id)}
                      title={track.locked ? "Unlock Track" : "Lock Track"}
                    >
                      {track.locked ? <Lock className="w-3 h-3" /> : <Unlock className="w-3 h-3" />}
                    </button>
                  </div>
                </div>

                {track.expanded &&
                  track.channels?.map((channel) => (
                    <div key={channel.id} className="track-sub-item">
                      <div className="subtrack-indent" />
                      <span
                        className="subtrack-color-badge"
                        style={{ backgroundColor: channel.color }}
                      />
                      <span className="subtrack-name">{channel.name}</span>
                    </div>
                  ))}
              </React.Fragment>
            ))}
          </div>
        </div>

        <div className="timeline-grid-container" ref={gridContainerRef}>
          <div className="timeline-grid-content" style={{ width: `${contentWidth}px` }}>
            <div
              className="timeline-ruler"
              ref={rulerRef}
              onMouseDown={handleRulerMouseDown}
            >
              {ticks.map(({ frame, isMajor }) => (
                <div
                  key={frame}
                  className={`ruler-tick ${isMajor ? "major" : "minor"}`}
                  style={{ left: `${(frame - startFrame) * zoom}px` }}
                >
                  {isMajor && <span className="tick-label">{frame}</span>}
                </div>
              ))}

              <div
                className="ruler-playhead-pill"
                style={{ left: `${playheadPositionX}px` }}
              >
                <span>{currentFrame}</span>
              </div>
            </div>

            <div
              className="timeline-playhead-line"
              style={{ left: `${playheadPositionX}px` }}
            />

            <div className="timeline-tracks-grid">
              {tracks.map((track) => (
                <React.Fragment key={track.id}>
                  <div
                    className={`grid-track-row ${track.type} ${
                      selectedBoneId === track.id ? "row-selected" : ""
                    }`}
                    onClick={(e) => {
                      const frame = getFrameFromMouseX(e.clientX);
                      setCurrentFrame(frame);
                    }}
                    onDoubleClick={(e) => {
                      const frame = getFrameFromMouseX(e.clientX);
                      addKeyframe(track.id, frame);
                    }}
                  >
                    {track.keyframes.map((kf) => {
                      const isSelected =
                        selectedKeyframe?.trackId === track.id &&
                        selectedKeyframe?.frame === kf.frame;

                      return (
                        <div
                          key={kf.frame}
                          className={`keyframe-diamond ${track.type} ${
                            isSelected ? "kf-selected" : ""
                          }`}
                          style={{
                            left: `${(kf.frame - startFrame) * zoom}px`,
                            backgroundColor:
                              track.type === "summary"
                                ? SUMMARY_TRACK_COLOR
                                : isSelected
                                ? SELECTED_KEYFRAME_COLOR
                                : track.color || DEFAULT_BONE_COLOR,
                          }}
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedKeyframe({
                              trackId: track.id,
                              frame: kf.frame,
                            });
                            setCurrentFrame(kf.frame);
                          }}
                          title={`${track.name} [Frame ${kf.frame}]`}
                        />
                      );
                    })}
                  </div>

                  {track.expanded &&
                    track.channels?.map((channel) => (
                      <div
                        key={channel.id}
                        className="grid-track-row subchannel-row"
                        onClick={(e) => {
                          const frame = getFrameFromMouseX(e.clientX);
                          setCurrentFrame(frame);
                        }}
                      >
                        {channel.keyframes.map((kf) => (
                          <div
                            key={kf.frame}
                            className="keyframe-diamond channel-kf"
                            style={{
                              left: `${(kf.frame - startFrame) * zoom}px`,
                              backgroundColor: channel.color,
                            }}
                            onClick={(e) => {
                              e.stopPropagation();
                              setCurrentFrame(kf.frame);
                            }}
                            title={`${channel.name} [Frame ${kf.frame}]`}
                          />
                        ))}
                      </div>
                    ))}
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Timeline;
