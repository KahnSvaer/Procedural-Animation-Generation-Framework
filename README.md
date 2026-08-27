# Procedural-Animation-Generation-Framework

## Milestones

### Milestone 0: 3D Foundational Model import to Unity Application
Demonstrates the import of 3D models into Unity applications.

<video src="assets/demo_videos/FrontEndDemoCompleteStaticSite.mp4" controls="controls" width="100%"></video>

### Milestone 1: Point to Spline to Armature Generation
Demonstrates the extraction and conversion of discrete spatial point clusters into continuous centripetal Catmull-Rom splines and hierarchical skeletal armatures.

<video src="assets/demo_videos/PointToSplineToArmatureDemo.mp4" controls="controls" width="100%"></video>

---

### Milestone 2: Morphological Part Segmentation
Demonstrates zero-shot vision foundation model (SAM3) and Shape Diameter Function (SDF) segmentation to identify and partition anatomical mesh appendages (dorsal fins, caudal tails).

#### 2.1 Dorsal Fin Segmentation
<video src="assets/demo_videos/Mesh-Segmentation-Demo/Dorsal-Fin-Segmentation.mp4" controls="controls" width="100%"></video>

#### 2.2 Tail / Caudal Fin Segmentation
<video src="assets/demo_videos/Mesh-Segmentation-Demo/Tail-Segmentation.mp4" controls="controls" width="100%"></video>

---

### Milestone 3: Procedural Wave Animation & Mesh Skinning
Demonstrates procedural wave kinematics (travelling and standing wave generators) driving skeletal chains with real-time Dual Quaternion Skinning (DQS) mesh deformation without any coupling with blender.

<video src="assets/demo_videos/Animation_Cylinder_Demo.mp4" controls="controls" width="100%"></video>

---

## Visualization
Development and visualization of 3D assets and animations is supported by [3D Mesh Viewer](https://marketplace.visualstudio.com/items?itemName=AssetToolLabs.mesh-viewer-vscode), a VS Code extension by AssetToolLabs.

## Data Source / References
Species images used for testing and mesh reconstruction are sourced from the [NOAA Fisheries](https://www.fisheries.noaa.gov/) species directory.
