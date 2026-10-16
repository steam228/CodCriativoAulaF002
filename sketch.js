// Pose grid: search photos online, detect the people in them with ml5 (MoveNet),
// and replace each person by an extruded 3D quad built from their wrists and ankles.
// The quads rotate with the mouse. See readme.md for the full walkthrough.

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------
const QUERY = "holding pisa tower"; // image search term
const ROWS = 5; // grid rows; columns are calculated from the window width
const RATIO = 3 / 4; // target cell width / height (vertical photos)

const CONF = 0.1; // min keypoint confidence; low on purpose so off-photo joints still get drawn

// Stickman bones as pairs of MoveNet keypoint names.
// Currently unused: the stickman drawing is commented out in drawStickman().
const BONES = [
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
];

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let imgs = []; // loaded p5.Image objects (only used as detector input and for crop maths)
let poses = []; // poses[i] = people detected in imgs[i]
let cols = 0; // number of columns that fill the window width
let cellW = 0; // cell size in pixels
let cellH = 0;

let loaded = false; // true once every image page has finished loading
let detected = 0; // how many images the detector has processed so far
let queue = Promise.resolve(); // detections run one at a time, in arrival order

let statusEl; // DOM element showing loading progress

// ---------------------------------------------------------------------------
// Setup: build the grid, then load images and run detection in parallel
// ---------------------------------------------------------------------------
async function setup() {
  createCanvas(windowWidth, windowHeight, WEBGL);

  // WEBGL text needs a loaded font, so progress is shown in a DOM element instead
  statusEl = createDiv("loading...");
  statusEl.style("position", "fixed");
  statusEl.style("left", "8px");
  statusEl.style("bottom", "6px");
  statusEl.style("color", "#888");
  statusEl.style("font", "12px sans-serif");

  computeGrid();

  const needed = ROWS * cols; // one image per grid cell
  const modelP = loadPoseModel(); // loads while images are downloading

  // Called for every image as soon as it has loaded: store it and queue its detection.
  const addImage = (img) => {
    if (imgs.length >= needed) return; // grid is already full
    const i = imgs.length;
    imgs.push(img);
    poses.push([]);
    redraw();
    // chain onto the queue so the model processes one image at a time
    queue = queue.then(async () => {
      poses[i] = await detectPoses(await modelP, img);
      detected++;
      redraw();
    });
  };

  // Fetch all search pages in parallel (20 results per page, one spare page as a buffer).
  const pages = Array.from({ length: min(7, ceil(needed / 20) + 1) }, (_, n) => n + 1);
  await Promise.all(pages.map((page) => loadPage(QUERY, page, addImage)));
  loaded = true;
  redraw();
}

// Load ml5's MoveNet in multi-person mode and resolve once the model is ready.
function loadPoseModel() {
  return new Promise((resolve) => {
    const m = ml5.bodyPose(
      "MoveNet",
      { modelType: "MULTIPOSE_LIGHTNING", enableSmoothing: false, minPoseScore: 0.2 },
      () => resolve(m)
    );
  });
}

// Promise wrapper around the callback-style model.detect().
// (Not named detect/loadModel: p5 2.x already has a global loadModel.)
function detectPoses(model, img) {
  return new Promise((resolve) => model.detect(img, resolve));
}

// Pick a column count so cells are close to RATIO, then stretch them to fill the width exactly.
function computeGrid() {
  cellH = windowHeight / ROWS;
  cols = max(1, round(windowWidth / (cellH * RATIO)));
  cellW = windowWidth / cols;
}

// ---------------------------------------------------------------------------
// Image search (Openverse, no API key needed)
// ---------------------------------------------------------------------------
// Fetch one page of results for tall images; each image is passed to onImage once loaded.
async function loadPage(query, page, onImage) {
  const url =
    "https://api.openverse.org/v1/images/?q=" +
    encodeURIComponent(query) +
    "&aspect_ratio=tall&page_size=20&page=" +
    page;
  let results = [];
  try {
    const res = await fetch(url);
    if (!res.ok) return;
    results = (await res.json()).results;
  } catch (e) {
    console.error(e);
    return;
  }
  // allSettled: an image that fails to load is skipped instead of breaking the page
  await Promise.allSettled(
    results.map((r) => loadImage(r.thumbnail || r.url).then(onImage))
  );
}

// ---------------------------------------------------------------------------
// Draw: runs every frame so the shapes follow the mouse
// ---------------------------------------------------------------------------
function draw() {
  background(0);
  const total = ROWS * cols;

  // loading progress, hidden once everything is analysed
  if (statusEl) {
    const busy = detected < imgs.length || !loaded;
    statusEl.html(busy ? `photos ${imgs.length}/${total} · analysed ${detected}` : "");
  }
  if (imgs.length === 0) return;

  // keep 2D-style coordinates (origin top-left) and no perspective distortion
  ortho();
  translate(-width / 2, -height / 2);

  // lights give the extruded faces their shading
  ambientLight(70);
  directionalLight(255, 255, 255, 0.4, 0.6, -1);
  directionalLight(120, 120, 160, -0.6, -0.2, -0.5);

  // shapes are not clipped to their cell, so partially visible bodies can extend past the photo
  for (let i = 0; i < total; i++) {
    if (!loaded && i >= imgs.length) break; // still filling the grid
    const k = i % imgs.length; // reuse images if there are fewer than cells
    const img = imgs[k];
    // cells fill column by column, top to bottom
    const x = floor(i / ROWS) * cellW;
    const y = (i % ROWS) * cellH;
    for (const person of poses[k]) drawStickman(person, img, x, y, cellW, cellH);
  }
}

// ---------------------------------------------------------------------------
// Pose helpers
// ---------------------------------------------------------------------------
// Map a keypoint from photo coordinates to canvas coordinates, using the same
// center-crop that fills a w x h cell with the photo (scale to cover, crop the overflow).
function toCell(kp, img, x, y, w, h) {
  const s = max(w / img.width, h / img.height); // cover scale
  const sx = (img.width - w / s) / 2; // cropped-away margin, in photo pixels
  const sy = (img.height - h / s) / 2;
  return createVector(x + (kp.x - sx) * s, y + (kp.y - sy) * s);
}

// Draw one detected person inside the cell at (x, y) of size w x h.
// Most of the original stickman code is commented out; only the extruded quad is active.
function drawStickman(person, img, x, y, w, h) {
  // keep confident keypoints only, indexed by name (e.g. pts.left_wrist)
  const pts = {};
  for (const kp of person.keypoints) {
    if (kp.confidence >= CONF) pts[kp.name] = toCell(kp, img, x, y, w, h);
  }

  // --- stickman (disabled) ---------------------------------------------------
  // stroke(255);
  // strokeWeight(max(2, w * 0.02));
  // strokeCap(ROUND);
  // noFill();

  // for (const [a, b] of BONES) {
  //   if (pts[a] && pts[b]) line(pts[a].x, pts[a].y, pts[b].x, pts[b].y);
  // }

  // // spine: mid-shoulders to mid-hips
  // const ls = pts.left_shoulder, rs = pts.right_shoulder;
  // const lh = pts.left_hip, rh = pts.right_hip;
  // let neck = null;
  // if (ls && rs) neck = p5.Vector.lerp(ls, rs, 0.5);
  // if (neck && lh && rh) {
  //   const pelvis = p5.Vector.lerp(lh, rh, 0.5);
  //   line(neck.x, neck.y, pelvis.x, pelvis.y);
  // }

  // // head: circle at the nose (or mid-eyes / mid-ears), sized from the face width or shoulders
  // const nose = pts.nose;
  // const eyes = pts.left_eye && pts.right_eye ? p5.Vector.lerp(pts.left_eye, pts.right_eye, 0.5) : null;
  // const center = nose || eyes;
  // if (center) {
  //   let r;
  //   if (pts.left_ear && pts.right_ear) r = pts.left_ear.dist(pts.right_ear) * 0.6;
  //   else if (pts.left_eye && pts.right_eye) r = pts.left_eye.dist(pts.right_eye) * 1.5;
  //   else if (ls && rs) r = ls.dist(rs) * 0.3;
  //   else r = w * 0.05;
  //   circle(center.x, center.y, r * 2);
  //   if (neck) line(center.x, center.y + r, neck.x, neck.y);
  // }

  // // red dots on wrists and ankles
  // noStroke();
  // fill(255, 0, 0);
  // for (const name of ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]) {
  //   if (pts[name]) circle(pts[name].x, pts[name].y, max(6, w * 0.06));
  // }

  // --- wrists + ankles quad (active) ------------------------------------------
  // Order matters: left wrist -> right wrist -> right ankle -> left ankle walks around
  // the shape, so the quad doesn't cross itself into a bow-tie for a normal standing pose.
  const quad4 = ["left_wrist", "right_wrist", "right_ankle", "left_ankle"].map((n) => pts[n]);
  // only when all four joints were detected
  if (quad4.every(Boolean)) drawExtrudedQuad(quad4, w * 0.25);
}

// ---------------------------------------------------------------------------
// 3D shape
// ---------------------------------------------------------------------------
// Extrude the 2D quad p (4 vectors) along z by `depth` (centered on z = 0)
// and rotate it around its own center with the mouse.
function drawExtrudedQuad(p, depth) {
  // center of the quad, used as the pivot for rotation
  const c = createVector(
    (p[0].x + p[1].x + p[2].x + p[3].x) / 4,
    (p[0].y + p[1].y + p[2].y + p[3].y) / 4
  );
  // vertices relative to the center, so rotation happens around it
  const v = p.map((q) => createVector(q.x - c.x, q.y - c.y));
  const hz = depth / 2; // half depth: front face at +hz, back face at -hz

  push();
  translate(c.x, c.y, 0);
  // mouse X spins around the vertical axis, mouse Y tilts around the horizontal one
  rotateY(map(mouseX, 0, width, -PI, PI));
  rotateX(map(mouseY, 0, height, PI, -PI));

  noStroke();
  fill(235);

  beginShape(TRIANGLES);
  // front and back faces: the quad is split into two triangles (0,1,2) and (0,2,3)
  for (const [z, nz] of [[hz, 1], [-hz, -1]]) {
    normal(0, 0, nz); // normal tells the lights which way the face points
    for (const i of [0, 1, 2, 0, 2, 3]) vertex(v[i].x, v[i].y, z);
  }
  // side faces: one rectangle (two triangles) per edge of the quad
  for (let i = 0; i < 4; i++) {
    const a = v[i], b = v[(i + 1) % 4];
    // edge normal in the xy plane, flipped if needed so it points away from the center
    let nx = b.y - a.y, ny = -(b.x - a.x);
    if (nx * (a.x + b.x) + ny * (a.y + b.y) < 0) { nx = -nx; ny = -ny; }
    const len = Math.hypot(nx, ny) || 1;
    normal(nx / len, ny / len, 0);
    for (const [q, z] of [[a, hz], [b, hz], [b, -hz], [a, hz], [b, -hz], [a, -hz]]) {
      vertex(q.x, q.y, z);
    }
  }
  endShape();
  pop();
}

// Recompute the grid when the window changes size (images are not re-fetched).
function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  computeGrid();
  redraw();
}
