# ImgKIT Architecture

## Overview
ImgKIT is a lightweight image manipulation tool built using Python and Streamlit. The system follows a fixed sequential image-processing pipeline.

## SDLC Workflow Diagram

![ImgKIT SDLC Flowchart](diagrams/imgkit_sdlc_flowchart.png)

Pipeline:

Upload → Metadata Extraction → Processing Pipeline → Visualization → Export

---

# Core Components

## 1. User Interface Layer
Handled using Streamlit.

Responsibilities:
- File upload
- Sliders and controls
- Multi-output configuration
- Histogram toggles
- Download buttons

---

## 2. Image Loading Layer

Libraries:
- PIL/Pillow
- rasterio

Responsibilities:
- Load JPG/PNG/TIFF images
- Handle multi-band GeoTIFFs
- Normalize raster bands
- Convert arrays into displayable RGB images

---

## 3. Metadata Layer

Extracts:
- CRS
- Latitude/Longitude
- Resolution
- Dimensions
- Channels
- Bit depth related information

Main library:
- rasterio

---

## 4. Processing Pipeline

Fixed pipeline order:

Crop → Greyscale → Blur → Brightness → Contrast → Resample

Libraries:
- scikit-image
- PIL
- NumPy

---

## 5. Visualization Layer

Responsibilities:
- Display original image
- Display processed outputs
- Histogram rendering
- RGB channel analysis

Library:
- matplotlib

---

## 6. Export Layer

Supports:
- PNG
- JPG
- TIFF

Includes:
- Bit depth conversion
- Multi-output download

---

# Workflow Design

ImgKIT follows a modular processing approach where each operation is isolated into dedicated functions.

Advantages:
- Easier debugging
- Better maintainability
- Flexible feature upgrades
- Improved readability

---

# Current Versions

- v1.0 → Core pipeline
- v2.0 → Histogram + resampling + bit depth
- v2.1 → Multi-output architecture

---

# Future Scope

- Histogram equalization
- Edge detection
- Sharpening filters
- Undo/redo workflow
- Plugin-based architecture
- GPU acceleration
