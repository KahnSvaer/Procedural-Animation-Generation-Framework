import React, { useEffect, useRef, useState, useCallback } from "react";
import { Vector3, Spherical } from "three";
import { useCameraStore } from "../../stores/cameraStore";
import "./Gizmo.css";

interface AxisDefinition {
  id: string;
  label: string;
  vector: Vector3;
  color: string;
  dimColor: string;
  isPositive: boolean;
}

const AXES: AxisDefinition[] = [
  {
    id: "pos-x",
    label: "X",
    vector: new Vector3(1, 0, 0),
    color: "#ef4444",
    dimColor: "#7f1d1d",
    isPositive: true,
  },
  {
    id: "neg-x",
    label: "-X",
    vector: new Vector3(-1, 0, 0),
    color: "#ef4444",
    dimColor: "#991b1b",
    isPositive: false,
  },
  {
    id: "pos-y",
    label: "Y",
    vector: new Vector3(0, 1, 0),
    color: "#22c55e",
    dimColor: "#14532d",
    isPositive: true,
  },
  {
    id: "neg-y",
    label: "-Y",
    vector: new Vector3(0, -1, 0),
    color: "#22c55e",
    dimColor: "#166534",
    isPositive: false,
  },
  {
    id: "pos-z",
    label: "Z",
    vector: new Vector3(0, 0, 1),
    color: "#3b82f6",
    dimColor: "#1e3a8a",
    isPositive: true,
  },
  {
    id: "neg-z",
    label: "-Z",
    vector: new Vector3(0, 0, -1),
    color: "#3b82f6",
    dimColor: "#1e40af",
    isPositive: false,
  },
];

const SIZE = 104;
const CENTER = SIZE / 2;
const RADIUS = 32;
const YAW_RING_RADIUS = 44;

interface ProjectedAxis extends AxisDefinition {
  x: number;
  y: number;
  z: number;
}

export const Gizmo: React.FC = () => {
  const camera = useCameraStore((state) => state.camera);
  const controls = useCameraStore((state) => state.controls);

  const containerRef = useRef<HTMLDivElement>(null);
  const [projectedAxes, setProjectedAxes] = useState<ProjectedAxis[]>([]);
  const [isYawDragging, setIsYawDragging] = useState(false);
  const [isOrbitDragging, setIsOrbitDragging] = useState(false);

  const dragModeRef = useRef<"none" | "yaw" | "orbit">("none");
  const lastPointerAngleRef = useRef(0);
  const lastPointerPosRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const animFrameIdRef = useRef<number | null>(null);
  const snapAnimRef = useRef<number | null>(null);

  const updateProjection = useCallback(() => {
    if (!camera) return;

    camera.updateMatrixWorld();
    const invMat = camera.matrixWorldInverse;

    const projected = AXES.map((axis) => {
      const u = axis.vector.clone().transformDirection(invMat);
      return {
        ...axis,
        x: CENTER + u.x * RADIUS,
        y: CENTER - u.y * RADIUS,
        z: u.z,
      };
    });

    projected.sort((a, b) => a.z - b.z);
    setProjectedAxes(projected);
  }, [camera]);

  useEffect(() => {
    let active = true;

    const loop = () => {
      if (!active) return;
      updateProjection();
      animFrameIdRef.current = requestAnimationFrame(loop);
    };

    loop();

    return () => {
      active = false;
      if (animFrameIdRef.current !== null) {
        cancelAnimationFrame(animFrameIdRef.current);
      }
      if (snapAnimRef.current !== null) {
        cancelAnimationFrame(snapAnimRef.current);
      }
    };
  }, [updateProjection]);

  const snapToAxis = useCallback(
    (axisVector: Vector3) => {
      if (!camera) return;

      if (snapAnimRef.current !== null) {
        cancelAnimationFrame(snapAnimRef.current);
        snapAnimRef.current = null;
      }

      const target = controls ? controls.target.clone() : new Vector3(0, 0, 0);
      const dist = Math.max(1, camera.position.distanceTo(target));
      const startPos = camera.position.clone();

      const currentDir = camera.position.clone().sub(target).normalize();
      const isCurrentlyFocused = currentDir.dot(axisVector) > 0.92;
      const effectiveVector = isCurrentlyFocused
        ? axisVector.clone().negate()
        : axisVector.clone();

      const endVector = effectiveVector.clone();
      if (Math.abs(endVector.x) < 0.0001 && Math.abs(endVector.z) < 0.0001) {
        endVector.z = 0.0001 * (endVector.y >= 0 ? 1 : -1);
      }
      const endPos = target.clone().add(endVector.clone().multiplyScalar(dist));

      const startUp = camera.up.clone();
      const endUp = new Vector3(0, 1, 0);

      const startTime = performance.now();
      const duration = 280;

      const animate = (now: number) => {
        const elapsed = now - startTime;
        const progress = Math.min(1, elapsed / duration);
        const ease = 1 - Math.pow(1 - progress, 3);

        camera.position.lerpVectors(startPos, endPos, ease);
        camera.up.lerpVectors(startUp, endUp, ease);
        camera.lookAt(target);

        if (controls) {
          controls.update();
        }

        updateProjection();

        if (progress < 1) {
          snapAnimRef.current = requestAnimationFrame(animate);
        } else {
          snapAnimRef.current = null;
        }
      };

      snapAnimRef.current = requestAnimationFrame(animate);
    },
    [camera, controls, updateProjection]
  );

  const getPointerAngle = (clientX: number, clientY: number): number => {
    if (!containerRef.current) return 0;
    const rect = containerRef.current.getBoundingClientRect();
    const cx = rect.left + CENTER;
    const cy = rect.top + CENTER;
    return Math.atan2(clientY - cy, clientX - cx);
  };

  const handleYawPointerDown = (e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    (e.target as Element).setPointerCapture(e.pointerId);

    dragModeRef.current = "yaw";
    setIsYawDragging(true);
    lastPointerAngleRef.current = getPointerAngle(e.clientX, e.clientY);
  };

  const handleOrbitPointerDown = (e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    (e.target as Element).setPointerCapture(e.pointerId);

    dragModeRef.current = "orbit";
    setIsOrbitDragging(true);
    lastPointerPosRef.current = { x: e.clientX, y: e.clientY };
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!camera) return;
    const target = controls ? controls.target.clone() : new Vector3(0, 0, 0);

    if (dragModeRef.current === "yaw") {
      const currentAngle = getPointerAngle(e.clientX, e.clientY);
      let deltaAngle = currentAngle - lastPointerAngleRef.current;

      if (deltaAngle > Math.PI) deltaAngle -= 2 * Math.PI;
      if (deltaAngle < -Math.PI) deltaAngle += 2 * Math.PI;

      lastPointerAngleRef.current = currentAngle;

      const forward = new Vector3();
      camera.getWorldDirection(forward);
      camera.up.applyAxisAngle(forward, -deltaAngle);
      camera.lookAt(target);

      if (controls) {
        controls.update();
      }
      updateProjection();
    } else if (dragModeRef.current === "orbit") {
      const dx = e.clientX - lastPointerPosRef.current.x;
      const dy = e.clientY - lastPointerPosRef.current.y;
      lastPointerPosRef.current = { x: e.clientX, y: e.clientY };

      const offset = camera.position.clone().sub(target);
      const spherical = new Spherical().setFromVector3(offset);

      spherical.theta -= dx * 0.008;
      spherical.phi -= dy * 0.008;
      spherical.phi = Math.max(0.01, Math.min(Math.PI - 0.01, spherical.phi));
      spherical.makeSafe();

      offset.setFromSpherical(spherical);
      camera.position.copy(target).add(offset);
      camera.lookAt(target);

      if (controls) {
        controls.update();
      }
      updateProjection();
    }
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    try {
      (e.target as Element).releasePointerCapture(e.pointerId);
    } catch {}
    dragModeRef.current = "none";
    setIsYawDragging(false);
    setIsOrbitDragging(false);
  };

  return (
    <div ref={containerRef} className="gizmo-container">
      <svg
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        className="gizmo-svg"
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
      >
        <circle
          cx={CENTER}
          cy={CENTER}
          r={RADIUS + 4}
          className={`gizmo-bg-circle ${isOrbitDragging ? "active" : ""}`}
          onPointerDown={handleOrbitPointerDown}
        />

        <circle
          cx={CENTER}
          cy={CENTER}
          r={YAW_RING_RADIUS}
          className={`gizmo-yaw-ring ${isYawDragging ? "active" : ""}`}
        />

        <circle
          cx={CENTER}
          cy={CENTER}
          r={YAW_RING_RADIUS}
          className="gizmo-yaw-ring-hit"
          onPointerDown={handleYawPointerDown}
        />

        {projectedAxes.map((axis) => {
          if (!axis.isPositive) return null;
          return (
            <line
              key={`line-${axis.id}`}
              x1={CENTER}
              y1={CENTER}
              x2={axis.x}
              y2={axis.y}
              stroke={axis.color}
              className="gizmo-axis-line"
              opacity={axis.z >= -0.2 ? 1 : 0.4}
            />
          );
        })}

        {projectedAxes.map((axis) => {
          const isPos = axis.isPositive;
          const knobRadius = isPos ? 8.5 : 7;
          const knobFill = isPos
            ? axis.color
            : axis.z >= 0
            ? axis.dimColor
            : "#1e293b";

          return (
            <g
              key={`knob-${axis.id}`}
              className="gizmo-axis-knob"
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                snapToAxis(axis.vector);
              }}
            >
              <circle
                cx={axis.x}
                cy={axis.y}
                r={12}
                fill="transparent"
                style={{ cursor: "pointer" }}
              />
              <circle
                cx={axis.x}
                cy={axis.y}
                r={knobRadius}
                fill={knobFill}
                stroke={isPos ? "#ffffff" : axis.color}
                strokeWidth={isPos ? 0.75 : 1.2}
                opacity={isPos ? 1 : axis.z < 0 ? 0.7 : 0.95}
              />
              <text
                x={axis.x}
                y={axis.y}
                className={`gizmo-axis-text ${isPos ? "" : "negative"}`}
              >
                {axis.label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
};

export default Gizmo;
