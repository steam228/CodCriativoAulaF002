# Code explained, step by step

A detailed walkthrough of [sketch.js](sketch.js), in the order the code runs. For the overview and the story of how it was built, see [readme.md](readme.md).

## The big picture

```text
search photos ──► load images ──► detect people ──► map joints to cells ──► extrude + rotate
 (Openverse)      (loadImage)      (ml5 MoveNet)       (toCell)             (WEBGL, mouse)
```

Everything after the search happens per image, and several stages overlap in time:

```text
time ─────────────────────────────────────────────────────────────►
model download  ████████████
page 1 images      ███ ██ ███
page 2 images        ██ ███ ██
page 3 images           ███ ██
detection (queue)          ▇▇ ▇▇ ▇▇ ▇▇ ▇▇ ▇▇ ▇▇ ...
draw()          every frame, shows whatever is ready
```

## Step 0: the HTML

[index.html](index.html) loads three scripts before the sketch: p5.js 2.3.4, p5.sound (loaded by the template and unused here) and ml5.js 1.4.0. p5 calls `setup()` and `draw()` on its own once the page has loaded.

## Step 1: constants

```js
const CARDS = ["People holding", "People pretending to be", /* ... */];
const ROWS = 5;
const RATIO = 3 / 4;
const CONF = 0.1;
```

These are the values you are most likely to change:

- `CARDS` are the openings of the fill-in-the-blank cards shown in the opening pop-up. The text the player adds is the rest of the search key.
- `ROWS` is the number of grid rows.
- `RATIO` is the wanted cell width divided by height. Below 1 means vertical cells.
- `CONF` is the minimum confidence a keypoint needs to count. Keypoints the model can't see get low confidence, so a low value keeps its guesses for joints outside the photo.

`BONES` lists the joints a stickman connects. Nothing uses it right now because the stickman drawing is commented out.

## Step 2: state variables

```js
let imgs = [];     // loaded images
let poses = [];    // poses[i] = people found in imgs[i]
let cols, cellW, cellH;
let loaded = false;
let detected = 0;
let queue = Promise.resolve();
let statusEl;
```

`imgs` and `poses` are parallel arrays: `poses[3]` always belongs to `imgs[3]`. The images themselves are never drawn. They are only given to the detector, and their width and height are used for the crop maths.

`queue` is a promise that every detection job attaches itself to. That makes the jobs run one after another instead of all at once.

## Step 3: `setup()`

`setup` is `async` because it waits for network requests.

### 3a. Canvas and the opening card

```js
createCanvas(windowWidth, windowHeight, WEBGL);
searchKey = await askCard();
```

`askCard()` builds a pop-up out of plain DOM elements: a dark overlay with a white card. The card shows a random opening from `CARDS`, such as "People holding", followed by a text box. The pop-up has two buttons: "Another card" picks a different opening, and "Search" (or the Enter key) submits. An empty answer is not accepted.

`askCard()` returns a promise, and `setup()` waits for it with `await`. Nothing is searched or loaded until the player answers. The promise resolves with the opening plus the answer in lower case, for example `"people holding pisa tower"`, and that is stored in `searchKey` and used for every search page.

While the pop-up is open `cardChosen` is false, and `keyPressed()` returns early. This matters because the spacebar handler ends with `return false`, which would stop the player from typing spaces in the text box.

### 3a2. Canvas and status element

```js
createCanvas(windowWidth, windowHeight, WEBGL);
statusEl = createDiv("loading...");
```

The `WEBGL` argument gives a 3D canvas. Text in a WEBGL canvas needs a loaded font file, so the progress counter is a normal HTML `div` positioned over the canvas.

### 3b. Grid size

`computeGrid()` runs next. See step 5.

### 3c. Start the model early

```js
const needed = ROWS * cols;
const modelP = loadPoseModel();
```

`needed` is one image per cell. `modelP` is a promise for the model. Nothing waits for it yet, so the model downloads in the background while the photos download.

### 3d. `addImage`: what to do with each loaded photo

```js
const addImage = (img) => {
  if (imgs.length >= needed) return;
  const i = imgs.length;
  imgs.push(img);
  poses.push([]);
  redraw();
  queue = queue.then(async () => {
    poses[i] = await detectPoses(await modelP, img);
    detected++;
    redraw();
  });
};
```

1. If the grid is already full, extra images are ignored.
2. `i` is the slot this image takes. The image goes into `imgs` and an empty list into `poses`.
3. A detection job is added to `queue`. It waits for the previous job, waits for the model, runs detection, then stores the result in `poses[i]`.

This is the point where downloading and detecting overlap: detection starts for the first image while the later images are still downloading.

### 3e. Request all the pages at once

```js
const pages = Array.from({ length: min(7, ceil(needed / 20) + 1) }, (_, n) => n + 1);
await Promise.all(pages.map((page) => loadPage(searchKey, page, addImage)));
loaded = true;
```

Each search page has 20 results. The code asks for enough pages to fill the grid, plus one spare, and never more than 7. `Promise.all` waits until all pages have finished. Then `loaded` becomes true.

## Step 4: `loadPoseModel()` and `detectPoses()`

```js
const m = ml5.bodyPose("MoveNet",
  { modelType: "MULTIPOSE_LIGHTNING", enableSmoothing: false, minPoseScore: 0.2 },
  () => resolve(m));
```

- `MULTIPOSE_LIGHTNING` finds several people per image. `SINGLEPOSE_LIGHTNING` is faster but finds one.
- `enableSmoothing: false` is for video. These are separate still images.
- `minPoseScore: 0.2` drops very uncertain people.

ml5 reports back with callbacks. Both helpers wrap that in a promise so the rest of the code can use `await`.

A detection result is a list of people. Each person has `keypoints`, a list of 17 objects like `{ name: "left_wrist", x, y, confidence }`. The coordinates are in pixels of the image that was analysed.

These helpers are not called `loadModel` and `detect` because p5 2.x already has a global `loadModel` for 3D models. A name clash with it broke an earlier version.

## Step 5: `computeGrid()`

```js
cellH = windowHeight / ROWS;
cols = max(1, round(windowWidth / (cellH * RATIO)));
cellW = windowWidth / cols;
```

Worked example, with a 1440 × 900 window and 5 rows:

1. `cellH = 900 / 5 = 180`.
2. A 3:4 cell would be 135 wide, so `1440 / 135 = 10.67` and `round` gives `cols = 11`.
3. `cellW = 1440 / 11 = 130.9`.

The columns fill the width exactly and every cell is the same size. The final cell shape is close to 3:4 but not exactly.

## Step 6: `loadPage()`

```js
const url = "https://api.openverse.org/v1/images/?q=" + encodeURIComponent(query)
  + "&aspect_ratio=tall&page_size=20&page=" + page;
```

- `encodeURIComponent` turns the spaces in the query into a URL-safe form.
- `aspect_ratio=tall` asks only for vertical photos.
- The `fetch` and `res.json()` are inside a `try`, so a network error is logged and the function returns without crashing the sketch.
- Each result has a `thumbnail` (a smaller copy served by Openverse) and a full `url`. The code prefers the thumbnail because it loads faster and is enough for the detector.
- `Promise.allSettled` loads all images of the page in parallel. An image that fails is skipped. Each successful one goes to `addImage`.

## Step 7: `draw()`

p5 calls `draw` every frame, so the shapes follow the mouse.

1. **Clear.** `background(0)`.
2. **Progress counter.** It shows `photos x/total · analysed n` while anything is still loading, and is empty after.
3. **Nothing yet.** If there are no images, it stops here and shows a black canvas.
4. **Camera set-up.** `ortho()` removes perspective, so a shape does not look bigger when closer to the camera. `translate(-width / 2, -height / 2)` moves the origin from the center (the WEBGL default) to the top-left, so coordinates work like in a normal 2D sketch.
5. **Lights.** One ambient light and two directional lights. Without lights, the extruded faces would all be the same flat colour and the 3D form wouldn't be visible.
6. **Cell loop.**

```js
const k = i % imgs.length;
const x = floor(i / ROWS) * cellW;
const y = (i % ROWS) * cellH;
```

Cell `i` is in column `floor(i / ROWS)` and row `i % ROWS`, so cells fill column by column, top to bottom. `k` picks the image. It repeats images if there are fewer than cells. While loading, the loop stops at the first cell with no image yet.

For every person in that image it calls `drawStickman()`.

The shapes are not clipped to their cell. A person near the edge of the photo can have a shape that extends past it.

## Step 8: `toCell()`: from photo pixels to canvas pixels

The keypoints are in photo coordinates, but the cell shows only a crop of the photo. This function applies the same crop to a point.

```js
const s  = max(w / img.width, h / img.height);   // scale so the photo covers the cell
const sx = (img.width  - w / s) / 2;              // photo pixels cropped off the left
const sy = (img.height - h / s) / 2;              // photo pixels cropped off the top
return createVector(x + (kp.x - sx) * s, y + (kp.y - sy) * s);
```

Worked example: a 300 × 400 photo in a 130 × 180 cell.

1. `s = max(130 / 300, 180 / 400) = max(0.433, 0.45) = 0.45`.
2. The visible width of the photo is `130 / 0.45 = 289` pixels, so `sx = (300 - 289) / 2 = 5.5` pixels are cropped from each side. Nothing is cropped top and bottom, so `sy = 0`.
3. A wrist at photo position `(150, 200)` goes to `cellX + (150 - 5.5) * 0.45 = cellX + 65` on x and `cellY + 200 * 0.45 = cellY + 90` on y. That's the middle of the cell.

A joint that was outside the photo, like `x = -40`, gives a negative result and lands to the left of the cell. That is what lets the shape extend past the photo.

## Step 9: `drawStickman()`

```js
const pts = {};
for (const kp of person.keypoints) {
  if (kp.confidence >= CONF) pts[kp.name] = toCell(kp, img, x, y, w, h);
}
```

This builds `pts`, an object that has only the confident joints, already in canvas coordinates, so you can write `pts.left_wrist`. A joint below `CONF` is missing from the object.

After that come the commented-out parts (bones, spine, head, red dots). To bring one back, remove the `//` at the start of its lines.

The active part:

```js
const quad4 = ["left_wrist", "right_wrist", "right_ankle", "left_ankle"].map((n) => pts[n]);
if (quad4.every(Boolean)) drawExtrudedQuad(quad4, w * 0.25);
```

- The joints are listed so that they go around the shape: top left, top right, bottom right, bottom left. In this order the quad does not cross itself for a person standing normally.
- A missing joint gives `undefined`, so `every(Boolean)` is true only when all four were found. If not, nothing is drawn for that person.
- The depth is a quarter of the cell width.

## Step 10: `drawExtrudedQuad()`

This function takes the four canvas points `p` and a depth, and builds a solid.

### 10a. Pivot and relative vertices

```js
const c = createVector(average x, average y);
const v = p.map((q) => createVector(q.x - c.x, q.y - c.y));
```

`c` is the center of the quad. The vertices `v` are measured from `c`. Later the code moves the origin to `c` and rotates, so the shape turns around its own center and not around the corner of the canvas.

### 10b. Transform

```js
push();
translate(c.x, c.y, 0);
rotateY(map(mouseX, 0, width, -PI, PI));
rotateX(map(mouseY, 0, height, PI, -PI));
```

- `push()` and `pop()` keep this rotation from affecting other shapes.
- Mouse X spins the shape around the vertical axis. At the left edge the angle is −180°, in the middle it's 0°, and at the right edge it is +180°.
- Mouse Y tilts it. It's mapped from `PI` to `-PI` (reversed) so that moving the mouse up tilts the top away, which feels natural.
- All shapes use the same angles, so they turn together.

### 10c. Front and back faces

```js
beginShape(TRIANGLES);
for (const [z, nz] of [[hz, 1], [-hz, -1]]) {
  normal(0, 0, nz);
  for (const i of [0, 1, 2, 0, 2, 3]) vertex(v[i].x, v[i].y, z);
}
```

`hz` is half the depth, so the solid is centered on `z = 0`. The loop runs twice, once for the front (`z = +hz`) and once for the back (`z = -hz`). A quad is made of two triangles: corners 0-1-2 and 0-2-3. `normal()` sets the direction the face is pointing, which the lights need to shade it.

### 10d. Side faces

```js
const a = v[i], b = v[(i + 1) % 4];
let nx = b.y - a.y, ny = -(b.x - a.x);
if (nx * (a.x + b.x) + ny * (a.y + b.y) < 0) { nx = -nx; ny = -ny; }
```

Each side connects corner `a` to the next corner `b`, as a rectangle between z = `+hz` and `-hz`, made of two triangles.

To get the side's normal, the code takes the edge `(b − a)` and turns it 90° with `(dy, -dx)`. That can point inward or outward, depending on the order of the points. The `if` checks it against the direction from the center to the middle of the edge (the dot product). If it is negative, the normal points inward, so it's flipped. After that it is made unit length (`len`).

### 10e. Finish

`endShape()` draws everything, and `pop()` restores the transform.

A known limit: the front face is split along the same diagonal every time. For a convex quad that is right. For a concave quad, it can produce a wrong face.

## Step 10b: spacebar toggle, `drawPhotos()` and `keyPressed()`

```js
let showPhotos = false;
function keyPressed() {
  if (key === " ") { showPhotos = !showPhotos; return false; }
}
```

`keyPressed` runs once per key press. The spacebar flips `showPhotos`, and `return false` stops the browser from scrolling the page.

When `showPhotos` is true, `draw()` calls `drawPhotos()` before it sets up the lights. It draws each photo in its cell with the nine-argument form of `image()`: the cell rectangle as the destination and the centered crop (the same maths as `toCell()`) as the source. Depth testing is switched off while the photos are drawn and back on afterwards, so the 3D shapes always appear on top of the photos, even when they rotate towards the back. Drawing the photos before the lights keeps them at their natural colours.

## Step 11: `windowResized()`

```js
resizeCanvas(windowWidth, windowHeight);
computeGrid();
redraw();
```

The canvas is resized and the grid recomputed. The photos are not fetched again, and `poses` were computed for the original set, so if you make the window much wider, the images repeat to fill the extra columns. Reload the page to get a fresh set.

## Debugging tips

| Symptom | Likely cause |
|---|---|
| Stuck on `loading...` | The Openverse request failed. Check the Network tab. |
| Counter shows photos but `analysed 0` | The model is still downloading, or ml5 failed. Check the console. |
| Errors mentioning `loadModel` | A function in the sketch clashes with a p5 function name. |
| Many empty cells | The model found no person, or fewer than 4 confident wrist/ankle joints. Lower `CONF`. |
| Strange shapes | The model guessed joints for a body that is cut off. Raise `CONF`. |
| Shapes look flat and white | Lights are missing, or `normal()` calls were removed. |
