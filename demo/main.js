// Import from pkg/ directory (symlinked in dev, copied in production build)
// For CDN usage, you can import from:
//   import init, { ... } from 'https://unpkg.com/llic-wasm@latest/llic.js';
//   import init, { ... } from 'https://cdn.jsdelivr.net/npm/llic-wasm@latest/llic.js';
import init, { compress, decompress, build_info, get_prediction_residual } from './pkg/llic.js';

let wasmReady = false;
let currentGrayData = null;
let currentDecompressed = null;
let currentResidual = null;
let currentError = null;
let currentWidth = 0;
let currentHeight = 0;
let currentCompressed = null;
let currentImageFile = null;
let currentBlur = 0;
let currentQuality = 'lossless'; // 'lossless', 'very_high', 'high', 'medium', 'low'
let currentView = 'side-by-side';
let comparePosition = 0.5;

// Debug overlay state
let showDecompressed = true;
let debugOverlay = 'none'; // 'none', 'residual', 'error'

// Quality level display names and error limits
const qualityInfo = {
  lossless: { name: 'Lossless', error: 0 },
  very_high: { name: 'Very High (±2)', error: 2 },
  high: { name: 'High (±4)', error: 4 },
  medium: { name: 'Medium (±8)', error: 8 },
  low: { name: 'Low (±16)', error: 16 },
};

async function initWasm() {
  await init();
  wasmReady = true;
  document.getElementById('version').textContent = build_info();
}

function formatThroughput(bytes, ms) {
  const mbps = (bytes / 1024 / 1024) / (ms / 1000);
  return `${mbps.toFixed(1)} MB/s`;
}

function updateStat(id, value) {
  document.getElementById(id).textContent = value;
}

function toGrayscale(imageData) {
  const gray = new Uint8Array(imageData.width * imageData.height);
  const data = imageData.data;
  for (let i = 0; i < gray.length; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    // Luminance formula
    gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
  }
  return gray;
}

function grayscaleToImageData(gray, width, height) {
  const imageData = new ImageData(width, height);
  for (let i = 0; i < gray.length; i++) {
    const v = gray[i];
    imageData.data[i * 4] = v;
    imageData.data[i * 4 + 1] = v;
    imageData.data[i * 4 + 2] = v;
    imageData.data[i * 4 + 3] = 255;
  }
  return imageData;
}

function roundToMultipleOf4(value) {
  return Math.floor(value / 4) * 4;
}

async function processImage(file, blur = currentBlur) {
  if (!wasmReady) {
    alert('WASM not ready yet');
    return;
  }

  // Store for re-processing with different blur
  if (file !== currentImageFile) {
    currentImageFile = file;
  }

  const img = new Image();
  const url = URL.createObjectURL(file);

  img.onload = () => {
    URL.revokeObjectURL(url);

    // Round dimensions to multiple of 4 (LLIC requirement)
    const width = roundToMultipleOf4(img.width);
    const height = roundToMultipleOf4(img.height);

    if (width === 0 || height === 0) {
      alert('Image too small (must be at least 4x4)');
      return;
    }

    // Draw original with optional blur and get grayscale
    const originalCanvas = document.getElementById('originalCanvas');
    originalCanvas.width = width;
    originalCanvas.height = height;
    const origCtx = originalCanvas.getContext('2d');
    if (blur > 0) {
      origCtx.filter = `blur(${blur}px)`;
    } else {
      origCtx.filter = 'none';
    }
    origCtx.drawImage(img, 0, 0, width, height);
    origCtx.filter = 'none';
    const originalImageData = origCtx.getImageData(0, 0, width, height);
    const grayData = toGrayscale(originalImageData);

    // Show grayscale on original canvas
    origCtx.putImageData(grayscaleToImageData(grayData, width, height), 0, 0);

    updateStat('imageSize', `${width} x ${height}`);
    updateStat('originalSize', `${grayData.length.toLocaleString()} bytes`);

    // Compress with selected quality
    let compressed;
    const compressStart = performance.now();
    try {
      compressed = compress(grayData, width, height, currentQuality);
    } catch (e) {
      updateStat('compressedSize', `Error: ${e}`);
      return;
    }
    const compressTime = performance.now() - compressStart;

    updateStat('compressedSize', `${compressed.length.toLocaleString()} bytes`);
    updateStat('ratio', `${(grayData.length / compressed.length).toFixed(2)}x (${(100 * compressed.length / grayData.length).toFixed(1)}%)`);
    updateStat('compressTime', `${compressTime.toFixed(2)} ms`);

    // Decompress
    let decompressed;
    const decompressStart = performance.now();
    try {
      decompressed = decompress(compressed, width, height);
    } catch (e) {
      updateStat('decompressTime', `Error: ${e}`);
      return;
    }
    const decompressTime = performance.now() - decompressStart;

    updateStat('decompressTime', `${decompressTime.toFixed(2)} ms`);
    updateStat('throughput', `${formatThroughput(grayData.length, compressTime)} / ${formatThroughput(grayData.length, decompressTime)}`);

    // Store for benchmarking
    currentGrayData = grayData;
    currentWidth = width;
    currentHeight = height;
    currentCompressed = compressed;
    document.getElementById('benchBtn').style.display = 'block';

    // Store for compare view
    currentDecompressed = decompressed;

    // Compute debug data
    currentResidual = get_prediction_residual(grayData, width, height);

    // Compute reconstruction error in JS (scaled for visibility)
    currentError = new Uint8Array(grayData.length);
    for (let i = 0; i < grayData.length; i++) {
      // Scale error by 8x for visibility (max error 16 * 8 = 128)
      currentError[i] = Math.min(255, Math.abs(grayData[i] - decompressed[i]) * 8);
    }

    // Calculate similarity (1.0 = identical, lower = lossy difference)
    let totalError = 0;
    for (let i = 0; i < grayData.length; i++) {
      totalError += Math.abs(grayData[i] - decompressed[i]);
    }
    const similarity = 1 - (totalError / (grayData.length * 255));
    updateStat('similarity', similarity.toFixed(6));

    // Draw decompressed (with debug overlays if enabled)
    const decompCanvas = document.getElementById('decompressedCanvas');
    decompCanvas.width = width;
    decompCanvas.height = height;
    updateDecompressedCanvas();

    // Update labels and compare view
    updateQualityLabel();
    if (currentView === 'compare') {
      updateCompareCanvas();
    }
  };

  img.src = url;
}

// File input handling
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    processImage(file);
  }
});
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) processImage(file);
});

// Benchmark function
function runBenchmark() {
  if (!currentGrayData) return;

  const iterations = 10;
  const warmup = 2;
  const compressTimes = [];
  const decompressTimes = [];
  const results = document.getElementById('benchResults');
  results.innerHTML = 'Running benchmark...';

  // Use setTimeout to allow UI update
  setTimeout(() => {
    // Warmup + benchmark
    for (let i = 0; i < warmup + iterations; i++) {
      const t0 = performance.now();
      const compressed = compress(currentGrayData, currentWidth, currentHeight, currentQuality);
      const t1 = performance.now();
      decompress(compressed, currentWidth, currentHeight);
      const t2 = performance.now();

      if (i >= warmup) {
        compressTimes.push(t1 - t0);
        decompressTimes.push(t2 - t1);
      }
    }

    // Calculate stats
    compressTimes.sort((a, b) => a - b);
    decompressTimes.sort((a, b) => a - b);

    const median = arr => arr[Math.floor(arr.length / 2)];
    const min = arr => arr[0];
    const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;

    const size = currentGrayData.length;
    const compressMedian = median(compressTimes);
    const decompressMedian = median(decompressTimes);

    const qualityName = qualityInfo[currentQuality].name;
    results.innerHTML = `
      <strong>Benchmark (${qualityName}, ${iterations} iter):</strong><br>
      <table style="margin-top:0.5rem; font-family:monospace;">
        <tr><td>Compress:</td><td>min ${min(compressTimes).toFixed(2)}ms, median ${compressMedian.toFixed(2)}ms</td></tr>
        <tr><td>Decompress:</td><td>min ${min(decompressTimes).toFixed(2)}ms, median ${decompressMedian.toFixed(2)}ms</td></tr>
        <tr><td>Throughput:</td><td>${formatThroughput(size, compressMedian)} / ${formatThroughput(size, decompressMedian)}</td></tr>
      </table>
    `;
  }, 10);
}

document.getElementById('benchBtn').addEventListener('click', runBenchmark);

// Blur slider handler
const blurSlider = document.getElementById('blurSlider');
const blurValue = document.getElementById('blurValue');

blurSlider.addEventListener('input', () => {
  currentBlur = parseFloat(blurSlider.value);
  blurValue.textContent = currentBlur;

  // Re-process current image with new blur
  if (currentImageFile && wasmReady) {
    processImage(currentImageFile, currentBlur);
  }
});

// Quality selector handler
document.querySelectorAll('input[name="quality"]').forEach(radio => {
  radio.addEventListener('change', (e) => {
    currentQuality = e.target.value;
    updateQualityLabel();

    // Re-process current image with new quality
    if (currentImageFile && wasmReady) {
      processImage(currentImageFile, currentBlur);
    }
  });
});

// Update panel label based on quality
function updateQualityLabel() {
  const label = document.getElementById('outputLabel');
  if (label) {
    label.textContent = `LLIC ${qualityInfo[currentQuality].name}`;
  }
  const compareLabel = document.querySelector('.compare-label-right');
  if (compareLabel) {
    compareLabel.textContent = `LLIC ${qualityInfo[currentQuality].name}`;
  }
}

// Load a demo image by filename
async function loadDemoImage(filename) {
  try {
    const response = await fetch(`./${filename}`);
    const blob = await response.blob();
    const file = new File([blob], filename, { type: 'image/webp' });
    await processImage(file);

    // Update active button state
    document.querySelectorAll('.demo-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.image === filename);
    });
  } catch (e) {
    console.log('Demo image not available:', e);
  }
}

// Demo image button handlers
document.querySelectorAll('.demo-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    if (wasmReady) {
      loadDemoImage(btn.dataset.image);
    }
  });
});

// WebGL Compare view functionality
let gl = null;
let glProgram = null;
let originalTexture = null;
let decompressedTexture = null;
let residualTexture = null;
let errorTexture = null;

const vertexShaderSource = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  varying vec2 v_texCoord;
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

const fragmentShaderSource = `
  precision mediump float;
  uniform sampler2D u_original;
  uniform sampler2D u_decompressed;
  uniform sampler2D u_residual;
  uniform sampler2D u_error;
  uniform float u_splitPos;
  uniform bool u_showDecompressed;
  uniform int u_debugMode; // 0 = none, 1 = residual, 2 = error
  varying vec2 v_texCoord;

  // Magma colormap function
  vec3 magma(float t) {
    if (t < 0.25) {
      return mix(vec3(0.0, 0.0, 0.04), vec3(0.28, 0.08, 0.47), t * 4.0);
    } else if (t < 0.5) {
      return mix(vec3(0.28, 0.08, 0.47), vec3(0.72, 0.22, 0.51), (t - 0.25) * 4.0);
    } else if (t < 0.75) {
      return mix(vec3(0.72, 0.22, 0.51), vec3(0.99, 0.53, 0.38), (t - 0.5) * 4.0);
    } else {
      return mix(vec3(0.99, 0.53, 0.38), vec3(0.99, 0.99, 0.75), (t - 0.75) * 4.0);
    }
  }

  void main() {
    vec4 original = texture2D(u_original, v_texCoord);

    // Build the right side (composited debug view)
    vec3 color = vec3(0.0);

    if (u_showDecompressed) {
      float gray = texture2D(u_decompressed, v_texCoord).r;
      color = vec3(gray);
    }

    if (u_debugMode == 1) {
      // Residual: use absolute value, 0.5 = zero error
      float r = texture2D(u_residual, v_texCoord).r;
      float intensity = abs(r - 0.5) * 2.0;
      vec3 overlayColor = magma(intensity);
      float blendAmount = u_showDecompressed ? intensity : 1.0;
      color = mix(color, overlayColor, blendAmount);
    } else if (u_debugMode == 2) {
      // Error: already 0-1 scaled
      float intensity = texture2D(u_error, v_texCoord).r;
      vec3 overlayColor = magma(intensity);
      float blendAmount = u_showDecompressed ? intensity : 1.0;
      color = mix(color, overlayColor, blendAmount);
    }

    gl_FragColor = v_texCoord.x < u_splitPos ? original : vec4(color, 1.0);
  }
`;

function initWebGL() {
  const canvas = document.getElementById('compareCanvas');
  gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
  if (!gl) {
    console.error('WebGL not supported');
    return false;
  }

  // Compile shaders
  const vertexShader = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vertexShader, vertexShaderSource);
  gl.compileShader(vertexShader);

  const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fragmentShader, fragmentShaderSource);
  gl.compileShader(fragmentShader);

  // Create program
  glProgram = gl.createProgram();
  gl.attachShader(glProgram, vertexShader);
  gl.attachShader(glProgram, fragmentShader);
  gl.linkProgram(glProgram);
  gl.useProgram(glProgram);

  // Set up geometry (fullscreen quad)
  const positions = new Float32Array([
    -1, -1,  1, -1,  -1, 1,
    -1, 1,   1, -1,   1, 1
  ]);
  const posBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
  const posLoc = gl.getAttribLocation(glProgram, 'a_position');
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

  // Texture coordinates (flip Y for correct orientation)
  const texCoords = new Float32Array([
    0, 1,  1, 1,  0, 0,
    0, 0,  1, 1,  1, 0
  ]);
  const texBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, texBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);
  const texLoc = gl.getAttribLocation(glProgram, 'a_texCoord');
  gl.enableVertexAttribArray(texLoc);
  gl.vertexAttribPointer(texLoc, 2, gl.FLOAT, false, 0, 0);

  // Create textures
  originalTexture = gl.createTexture();
  decompressedTexture = gl.createTexture();
  residualTexture = gl.createTexture();
  errorTexture = gl.createTexture();

  return true;
}

function uploadGrayscaleTexture(texture, data, width, height, textureUnit) {
  gl.activeTexture(gl.TEXTURE0 + textureUnit);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, width, height, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, data);
}

function updateCompareCanvas() {
  if (!currentGrayData || !currentDecompressed) return;

  const canvas = document.getElementById('compareCanvas');

  // Initialize WebGL on first use or if context was lost
  if (!gl || gl.isContextLost()) {
    canvas.width = currentWidth;
    canvas.height = currentHeight;
    if (!initWebGL()) return;
  } else if (canvas.width !== currentWidth || canvas.height !== currentHeight) {
    canvas.width = currentWidth;
    canvas.height = currentHeight;
    gl.viewport(0, 0, currentWidth, currentHeight);
  }

  // Upload textures
  uploadGrayscaleTexture(originalTexture, currentGrayData, currentWidth, currentHeight, 0);
  uploadGrayscaleTexture(decompressedTexture, currentDecompressed, currentWidth, currentHeight, 1);
  if (currentResidual) {
    uploadGrayscaleTexture(residualTexture, currentResidual, currentWidth, currentHeight, 2);
  }
  if (currentError) {
    uploadGrayscaleTexture(errorTexture, currentError, currentWidth, currentHeight, 3);
  }

  // Set uniforms
  gl.useProgram(glProgram);
  gl.uniform1i(gl.getUniformLocation(glProgram, 'u_original'), 0);
  gl.uniform1i(gl.getUniformLocation(glProgram, 'u_decompressed'), 1);
  gl.uniform1i(gl.getUniformLocation(glProgram, 'u_residual'), 2);
  gl.uniform1i(gl.getUniformLocation(glProgram, 'u_error'), 3);
  gl.uniform1f(gl.getUniformLocation(glProgram, 'u_splitPos'), comparePosition);
  gl.uniform1i(gl.getUniformLocation(glProgram, 'u_showDecompressed'), showDecompressed ? 1 : 0);
  const debugMode = debugOverlay === 'residual' ? 1 : (debugOverlay === 'error' ? 2 : 0);
  gl.uniform1i(gl.getUniformLocation(glProgram, 'u_debugMode'), debugMode);

  // Draw
  gl.viewport(0, 0, currentWidth, currentHeight);
  gl.drawArrays(gl.TRIANGLES, 0, 6);

  // Update slider position
  const slider = document.getElementById('compareSlider');
  const canvasRect = canvas.getBoundingClientRect();
  slider.style.left = `${comparePosition * canvasRect.width}px`;
}

// View toggle handlers
document.querySelectorAll('.view-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    currentView = btn.dataset.view;
    document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const sideBySide = document.getElementById('sideBySideView');
    const compare = document.getElementById('compareView');

    if (currentView === 'side-by-side') {
      sideBySide.style.display = 'grid';
      compare.style.display = 'none';
    } else {
      sideBySide.style.display = 'none';
      compare.style.display = 'block';
      // Delay to ensure layout is complete before positioning slider
      requestAnimationFrame(() => updateCompareCanvas());
    }
  });
});

// Compare slider drag
const compareSlider = document.getElementById('compareSlider');
const compareWrapper = document.getElementById('compareWrapper');
let isDragging = false;

function updateSliderPosition(clientX) {
  const canvas = document.getElementById('compareCanvas');
  const rect = canvas.getBoundingClientRect();
  const x = clientX - rect.left;
  comparePosition = Math.max(0, Math.min(1, x / rect.width));

  // Fast path: just update the uniform and redraw (no texture re-upload)
  if (gl && glProgram && !gl.isContextLost()) {
    gl.useProgram(glProgram);
    gl.uniform1f(gl.getUniformLocation(glProgram, 'u_splitPos'), comparePosition);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // Update slider position
    const slider = document.getElementById('compareSlider');
    slider.style.left = `${comparePosition * rect.width}px`;
  } else {
    updateCompareCanvas();
  }
}

compareSlider.addEventListener('mousedown', (e) => {
  isDragging = true;
  e.preventDefault();
});

// Allow clicking anywhere on the canvas to reposition
document.getElementById('compareCanvas').addEventListener('click', (e) => {
  updateSliderPosition(e.clientX);
});

document.addEventListener('mousemove', (e) => {
  if (isDragging) {
    updateSliderPosition(e.clientX);
  }
});

document.addEventListener('mouseup', () => {
  isDragging = false;
});

// Touch support for compare slider
compareSlider.addEventListener('touchstart', (e) => {
  isDragging = true;
  e.preventDefault();
});

document.addEventListener('touchmove', (e) => {
  if (isDragging && e.touches.length > 0) {
    updateSliderPosition(e.touches[0].clientX);
  }
});

document.addEventListener('touchend', () => {
  isDragging = false;
});

// Debug overlay handlers
document.getElementById('showDecompressed').addEventListener('change', (e) => {
  showDecompressed = e.target.checked;
  updateDecompressedCanvas();
  if (currentView === 'compare') {
    updateCompareCanvas();
  }
});

document.querySelectorAll('input[name="debugOverlay"]').forEach(radio => {
  radio.addEventListener('change', (e) => {
    debugOverlay = e.target.value;
    updateDecompressedCanvas();
    if (currentView === 'compare') {
      updateCompareCanvas();
    }
  });
});

// Magma colormap function for JS
function magmaColor(t) {
  let r, g, b;
  if (t < 0.25) {
    const s = t * 4;
    r = s * 0.28 * 255;
    g = s * 0.08 * 255;
    b = (0.04 + s * 0.43) * 255;
  } else if (t < 0.5) {
    const s = (t - 0.25) * 4;
    r = (0.28 + s * 0.44) * 255;
    g = (0.08 + s * 0.14) * 255;
    b = (0.47 + s * 0.04) * 255;
  } else if (t < 0.75) {
    const s = (t - 0.5) * 4;
    r = (0.72 + s * 0.27) * 255;
    g = (0.22 + s * 0.31) * 255;
    b = (0.51 - s * 0.13) * 255;
  } else {
    const s = (t - 0.75) * 4;
    r = 255;
    g = (0.53 + s * 0.46) * 255;
    b = (0.38 + s * 0.37) * 255;
  }
  return { r, g, b };
}

// Update the decompressed canvas with debug overlays (side-by-side view)
function updateDecompressedCanvas() {
  if (!currentDecompressed || !currentWidth || !currentHeight) return;

  const canvas = document.getElementById('decompressedCanvas');
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(currentWidth, currentHeight);

  for (let i = 0; i < currentDecompressed.length; i++) {
    let r = 0, g = 0, b = 0;

    if (showDecompressed) {
      const gray = currentDecompressed[i];
      r = gray;
      g = gray;
      b = gray;
    }

    if (debugOverlay !== 'none') {
      let intensity = 0;
      if (debugOverlay === 'residual' && currentResidual) {
        const res = currentResidual[i] / 255;
        intensity = Math.abs(res - 0.5) * 2;
      } else if (debugOverlay === 'error' && currentError) {
        intensity = currentError[i] / 255;
      }

      const overlay = magmaColor(intensity);
      const blendAmount = showDecompressed ? intensity : 1.0;
      r = r * (1 - blendAmount) + overlay.r * blendAmount;
      g = g * (1 - blendAmount) + overlay.g * blendAmount;
      b = b * (1 - blendAmount) + overlay.b * blendAmount;
    }

    imageData.data[i * 4] = Math.round(r);
    imageData.data[i * 4 + 1] = Math.round(g);
    imageData.data[i * 4 + 2] = Math.round(b);
    imageData.data[i * 4 + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);
}

// Hover info display
function updateHoverInfo(x, y) {
  const hoverInfo = document.getElementById('hoverInfo');
  if (!currentGrayData || x < 0 || y < 0 || x >= currentWidth || y >= currentHeight) {
    hoverInfo.innerHTML = `
      <span class="hover-coords">&nbsp;</span>
      <div class="hover-values">
        <div class="hover-row"><span>Original:</span><span>-</span></div>
        <div class="hover-row"><span>Decompressed:</span><span>-</span></div>
        <div class="hover-row"><span>Residual:</span><span>-</span></div>
        <div class="hover-row"><span>Error:</span><span>-</span></div>
      </div>`;
    return;
  }

  const idx = y * currentWidth + x;
  const original = currentGrayData[idx];
  const decompressed = currentDecompressed ? currentDecompressed[idx] : '-';
  const residual = currentResidual ? currentResidual[idx] : null;
  const error = currentDecompressed ? Math.abs(currentGrayData[idx] - currentDecompressed[idx]) : 0;

  // Convert residual from 0-255 (128=zero) to signed value
  const residualSigned = residual !== null ? residual - 128 : null;
  const residualStr = residualSigned !== null ? `${residualSigned >= 0 ? '+' : ''}${residualSigned}` : '-';
  const errorStr = error > 0 ? `±${error}` : '0';

  hoverInfo.innerHTML = `
    <span class="hover-coords">(${x}, ${y})</span>
    <div class="hover-values">
      <div class="hover-row"><span>Original:</span><span>${original}</span></div>
      <div class="hover-row"><span>Decompressed:</span><span>${decompressed}</span></div>
      <div class="hover-row"><span>Residual:</span><span>${residualStr}</span></div>
      <div class="hover-row"><span>Error:</span><span>${errorStr}</span></div>
    </div>`;
}

function getCanvasPixelCoords(canvas, clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = Math.floor((clientX - rect.left) * scaleX);
  const y = Math.floor((clientY - rect.top) * scaleY);
  return { x, y };
}

// Add hover listeners to both canvases
['decompressedCanvas', 'originalCanvas', 'compareCanvas'].forEach(id => {
  const canvas = document.getElementById(id);
  if (canvas) {
    canvas.addEventListener('mousemove', (e) => {
      const { x, y } = getCanvasPixelCoords(canvas, e.clientX, e.clientY);
      updateHoverInfo(x, y);
    });
    canvas.addEventListener('mouseleave', () => {
      updateHoverInfo(-1, -1);
    });
  }
});

// Initialize and load first demo image
initWasm().then(() => {
  loadDemoImage('demo_image2.webp');
}).catch(console.error);
