# SAM3 Backend Investigation

Source Model: https://huggingface.co/facebook/sam3

## Overview

| Name | Link | Backend | Quantization | Size | File Name | Notes |
|--------|--------|--------|--------|--------|--------|--------|
| facebook/sam3 | https://huggingface.co/facebook/sam3 | PyTorch / Transformers | FP16/FP32 (Base) | Unknown | | Official Meta SAM3 |
| onnx-community/sam3-tracker-ONNX | https://huggingface.co/onnx-community/sam3-tracker-ONNX | ONNX | Unknown | Unknown | | Tracker-focused ONNX export |
| JohnJohnWong/sam3-tracker-ONNX | https://huggingface.co/JohnJohnWong/sam3-tracker-ONNX | ONNX | Unknown | Unknown | | Alternative tracker ONNX export |
| Luminia/sam3-v2-onnx | https://huggingface.co/Luminia/sam3-v2-onnx | ONNX Runtime | UINT8 Dynamic | ~887 MB | | Text-prompt segmentation export |
| Kishanstar2003/SAM3_ONNX_FP16 | https://huggingface.co/Kishanstar2003/SAM3_ONNX_FP16 | ONNX Runtime | FP16 | Unknown | | FP16 ONNX export |
| embedl/sam3 | https://huggingface.co/embedl/sam3 | TensorRT | INT8 + FP16 Mixed Precision | ~3.1 GB+ | | NVIDIA optimized |
| mlx-community/sam3-8bit | https://huggingface.co/mlx-community/sam3-8bit | MLX | 8-bit | ~1.04 GB | | Includes video tracking |
| mlx-community/sam3-6bit | https://huggingface.co/mlx-community/sam3-6bit | MLX | 6-bit | ~0.2 GB | | Experimental |
| mlx-community/sam3-5bit | https://huggingface.co/mlx-community/sam3-5bit | MLX | 5-bit | ~0.2 GB | | Experimental |
| mlx-community/sam3-4bit | https://huggingface.co/mlx-community/sam3-4bit | MLX | 4-bit | ~0.2 GB | | Experimental |
| mlx-community/sam3-mxfp8 | https://huggingface.co/mlx-community/sam3-mxfp8 | MLX | MXFP8 | ~0.3 GB | | Experimental |
| mlx-community/sam3-mxfp4 | https://huggingface.co/mlx-community/sam3-mxfp4 | MLX | MXFP4 | ~0.2 GB | | Experimental |
| mlx-community/sam3-nvfp4 | https://huggingface.co/mlx-community/sam3-nvfp4 | MLX | NVFP4 | ~0.2 GB | | Experimental |
| manak0/sam3-5bit | https://huggingface.co/manak0/sam3-5bit | MLX | 5-bit | ~0.2 GB | | Community quantization |
| danilobukvic/sam3-text-onnx | https://huggingface.co/danilobukvic/sam3-text-onnx | ONNX | Unknown | Unknown | | Text-focused export |

---

## Candidate Backends

| Backend | Priority | Reason |
|----------|----------|----------|
| ONNX Runtime | High | Cross-platform, easiest integration |
| MLX | High | Apple Silicon acceleration |
| TensorRT | Medium | NVIDIA acceleration |
| PyTorch Reference | Low | Useful for validation only |

---

## Features Matrix

| Model | Image Segmentation | Point Prompt | Box Prompt | Text Prompt | Video Tracking | Notes |
|---------|---------|---------|---------|---------|---------|---------|
| facebook/sam3 | ? | ? | ? | ? | ? | Verify from docs |
| onnx-community/sam3-tracker-ONNX | ✓ | ✓ | ? | ? | ? | Investigate |
| mlx-community/sam3-8bit | ✓ | ? | ? | ✓ | ✓ | Confirm prompt support |
| Luminia/sam3-v2-onnx | ✓ | ? | ✓ | ✓ | ? | Appears text-first |
| embedl/sam3 | ✓ | ? | ? | ✓ | ✓ | TensorRT deployment |

---

## Experiment Notes

### facebook/sam3

**File Name:**  
-

**Observations:**  
-

---

### onnx-community/sam3-tracker-ONNX

**File Name:**  
-

**Observations:**  
-

---

### mlx-community/sam3-8bit

**File Name:**  
-

**Observations:**  
-

---

### Luminia/sam3-v2-onnx

**File Name:**  
-

**Observations:**  
-

---

### embedl/sam3

**File Name:**  
-

**Observations:**  
-



