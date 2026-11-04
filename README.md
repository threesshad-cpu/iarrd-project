# Astronomical Image Processing Prototype

A browser-based prototype for exploring an image-processing workflow for astronomical imagery. It demonstrates image upload, preprocessing, bright-object detection, and lightweight result storage in a React interface.

## Features

- Drag-and-drop image upload with file validation.
- Configurable Gaussian blur and detection threshold controls.
- Grayscale conversion, filtering, and normalization.
- Bright-spot clustering with centroid, area, and brightness summaries.
- Results view with JSON export and local saved-result management.
- Responsive interface with accessible controls and light/dark theme support.

## Technology

React · JavaScript · Image Processing · HTML · CSS · Local Storage

## Run locally

```bash
git clone https://github.com/threesshad-cpu/iarrd-project.git
cd iarrd-project
npm install
npm start
```

Open the local URL printed by the development server.

## Implementation scope and limitations

The current prototype performs preprocessing and lightweight bright-spot detection in the browser using JavaScript. It is not a trained CNN-based astronomical classifier. Production-grade processing would require validated astronomical datasets, FITS support, robust detection algorithms, and server-side model inference.

## Source

[View the repository](https://github.com/threesshad-cpu/iarrd-project)
