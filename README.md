# Astronomical UI — Enhanced Prototype

This repository is a small React prototype demonstrating a client-side pipeline for astronomical image preprocessing, enhancement, simple object detection and lightweight feature storage. It's intended as a UI/UX front end for the following problem statement:

Problem: Telescope images contain noise and distortions. Manual analysis is slow. An automated AI + image processing pipeline is needed to enhance images and identify celestial objects.

Solution: This UI demonstrates browser-side preprocessing (gaussian filtering, normalization), light-weight detection and saving extracted features to a local feature DB. It is a prototype for integrating server-side CNN models later.

Key features added
- Responsive, accessible `Navbar` with theme toggle (light/dark).
- `Upload` page: drag-and-drop, file validation (max 10MB), filename & dimensions, Gaussian sigma and detection threshold sliders.
- Client-side preprocessing: grayscale conversion, Gaussian blur (configurable), normalization.
- Lightweight detector: bright-spot clustering yielding centroid, area, brightness.
- `Results` page: shows enhanced image, detected objects, save to feature DB (localStorage), export JSON, view/delete saved entries with pagination.
- UX polish: processing spinner overlay, dropzone highlight, accessible buttons and labels.

How to run

1. Install dependencies:

```bash
npm install
```

2. Start development server:

```bash
npm start
```

Open http://localhost:3000 in your browser.

Developer notes
- This project runs preprocessing and a simple detector purely in JavaScript for demonstration. For production use you should run heavy image processing and CNN inference on a backend with GPUs and persist results to a database.
- Suggested next steps: support FITS files (astronomy standard), integrate a server-side CNN for classification, add user accounts and server-side DB, and advanced deblending algorithms.

Files you may want to inspect
- `src/pages/Upload.js` — drag-and-drop + preprocessing + detection pipeline.
- `src/pages/Results.js` — display enhanced image, detected objects, feature DB save/export.
- `src/components/Navbar.js` — responsive navbar and theme toggle.

If you want, I can:
- Integrate a lightweight backend to run a TensorFlow/PyTorch model and persist features.
- Add FITS support and metadata extraction.
- Improve detection with deblending and morphological operations.

Enjoy exploring!
### Code Splitting
