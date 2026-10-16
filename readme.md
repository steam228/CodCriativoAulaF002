# Pose grid

A p5.js sketch that searches the web for photos, detects the people in them with a machine-learning model, and replaces every person with an abstract 3D shape: a quad stretched between their **wrists and ankles**, extruded into a solid and rotated with the mouse.

Search term: you choose it. A pop-up shows a fill-in-the-blank card such as "People holding ___". Complete it, for example with "pisa tower", and the search key becomes "people holding pisa tower". The card list is `CARDS` in [sketch.js](sketch.js).

## Running it

1. Serve the folder over HTTP, for example with the VS Code Live Server extension or `npx serve`. Opening `index.html` as a `file://` page won't work for the image requests.
2. Open the page, complete a card in the pop-up and press Enter or click Search. "Another card" draws a different one. Then wait. The grey counter at the bottom left shows photos loaded and photos analysed. The first load also downloads the ml5 model, so it's slower than later loads.
3. Move the mouse to rotate all the shapes.
4. Press the spacebar to show or hide the photo grid behind the shapes. It starts hidden.

Libraries (loaded from a CDN in [index.html](index.html)): p5.js 2.3.4 and ml5.js 1.4.0. No API key is needed.

## How it was built

The sketch grew in small steps. Each step changed one thing:

1. **Fetch and show photos.** Web image search needs a source that allows it. Scraping TikTok or Instagram isn't possible without login, API approval or breaking their terms. [Openverse](https://openverse.org) has no key requirement, supports a tall-image filter, and returns openly licensed images.
2. **Make a grid.** Vertical photos, all cropped to the same cell size, with a fixed number of rows (10, then 8, 6 and finally 5). The column count is calculated, so the grid always fills the full window width.
3. **Detect people.** ml5's MoveNet finds each person's 17 body keypoints in every photo.
4. **Draw a stickman.** White lines on black instead of the photo. Joints the model places outside the photo are still drawn, so cropped bodies extend past their cell.
5. **Speed it up.** The first version loaded everything one step after another. Now the model downloads while photos download, and detection starts as soon as each photo arrives.
6. **Fix a name clash.** A helper called `loadModel` was replaced by p5's own 3D `loadModel`. It was renamed `loadPoseModel`.
7. **Simplify to four points.** Red dots on the wrists and ankles, then the stickman and dots were commented out and a 1 px quad was drawn between those four joints, only when all four were detected.
8. **Make it 3D.** The quad became an extruded solid in a WEBGL canvas, lit and rotated by the mouse.

## The code, step by step

### 1. Settings

```js
const CARDS = ["People holding", "People pretending to be", /* ... */];
const ROWS = 5;
const RATIO = 3 / 4;
const CONF = 0.1;
```

- `CARDS`: the openings of the fill-in-the-blank cards. The player's answer is added to the end to make the search key.
- `ROWS`: number of grid rows.
- `RATIO`: the cell width divided by its height, so cells are vertical.
- `CONF`: the minimum confidence a keypoint needs to be used. It is low on purpose. The model gives low confidence to joints it can't see, and a low threshold lets those guesses outside the photo still count.

`BONES` lists which keypoints a stickman connects. It's only used by the commented-out stickman code.

### 2. State

| Variable | What it holds |
|---|---|
| `imgs` | Loaded images. They're only used as detector input and for the crop maths. They are never drawn. |
| `poses` | For each image, the list of people the model found. |
| `cols`, `cellW`, `cellH` | The grid size, calculated in `computeGrid()`. |
| `loaded`, `detected` | Progress flags for the counter. |
| `queue` | A chain of promises so detections run one at a time. |

### 3. `computeGrid()`

```js
cellH = windowHeight / ROWS;
cols = max(1, round(windowWidth / (cellH * RATIO)));
cellW = windowWidth / cols;
```

The row height comes from the window height. The column count is the whole number of 3:4 cells closest to the window width. Then the cell width is stretched slightly so the columns fill the width exactly. All cells have the same size.

### 4. `setup()`

1. Creates a `WEBGL` canvas and a small HTML element for the progress counter. Text in WEBGL needs a loaded font, so the counter is plain HTML instead.
2. Calculates the grid and starts loading the model straight away (`loadPoseModel()`).
3. Defines `addImage`, which runs for every image as soon as it loads. It stores the image and adds a detection job to `queue`.
4. Requests all the search pages in parallel with `loadPage()`. It asks for enough pages to fill the grid, plus one spare.
5. Sets `loaded = true` when every page has finished.

### 5. `loadPage()`: image search

Calls the Openverse API with `aspect_ratio=tall` for vertical photos and `page_size=20`. For each result it loads the thumbnail with `loadImage()` and passes it to `addImage`. `Promise.allSettled` means one broken image link is skipped and doesn't stop the others.

### 6. `loadPoseModel()` and `detectPoses()`

ml5 uses callbacks, so both are wrapped in promises to work with `await`. The model is MoveNet in `MULTIPOSE_LIGHTNING` mode, which can find several people in one photo. It returns keypoints with `name`, `x`, `y` and `confidence`, in photo pixel coordinates.

### 7. `draw()`

Runs every frame, so the shapes follow the mouse.

1. Clears to black and updates the progress counter.
2. `ortho()` turns off perspective so shapes aren't distorted. `translate(-width / 2, -height / 2)` moves the origin to the top-left corner, so coordinates work like a normal 2D sketch.
3. Sets up three lights. Without them the solid faces would all look the same flat colour.
4. Loops over the grid cells. Cell `i` sits at column `floor(i / ROWS)` and row `i % ROWS`, so cells fill column by column. If there are fewer images than cells, images are reused (`i % imgs.length`).
5. For each person found in that cell's image, calls `drawStickman()`.

Nothing is clipped, so a shape can extend beyond its cell.

### 8. `toCell()`: photo coordinates to canvas coordinates

The photo is scaled to cover its cell and the overflow is cropped, like CSS `object-fit: cover`:

```js
const s = max(w / img.width, h / img.height);
```

`s` is the scale. The margin that gets cropped away is `(img.width - w / s) / 2` on x, and the same on y. A keypoint maps to the canvas as `cellX + (kp.x - cropMarginX) * s`. This puts every joint where it was in the cropped photo, even if it ends up outside the cell.

### 9. `drawStickman()`

1. Builds `pts`, an object with the confident keypoints converted to canvas coordinates, for example `pts.left_wrist`.
2. The stickman (bones, spine, head) and the red dots are commented out. Uncomment them to bring them back.
3. Collects the four points in the order left wrist, right wrist, right ankle, left ankle. That order goes around the shape, so a person standing normally gives a proper quad and not a bow-tie.
4. If all four exist, calls `drawExtrudedQuad()`. If any is missing, nothing is drawn for that person.

### 10. `drawExtrudedQuad()`: the 3D shape

1. **Pivot.** The center of the four points is the pivot. The vertices are stored relative to it, so rotation happens around the shape's own center.
2. **Rotation.** `translate()` moves to the center, then `rotateY` follows mouse X (−180° to 180°) and `rotateX` follows mouse Y.
3. **Faces.** The solid is built from triangles:
   - Front and back: the quad split into two triangles, at `z = +depth/2` and `z = -depth/2`.
   - Four sides: one rectangle per edge, made of two triangles.
4. **Normals.** `normal()` tells the lights which way each face points. For the side faces the normal is perpendicular to the edge and flipped if needed so it points away from the center.
5. **Depth** is `cellWidth * 0.25`, set in the call from `drawStickman()`.

The quad is split along one diagonal. That is correct for convex quads, but a very twisted or concave quad may show a wrong face.

### 11. `windowResized()`

Resizes the canvas and recomputes the grid. The photos aren't fetched again, so after making the window much wider, images are reused to fill the extra columns. Reload for a fresh set.

## Things to tweak

- `CARDS`, `EXAMPLE_ANSWER`: the card openings and the placeholder in the answer box. `ROWS`: how many rows.
- `CONF`: lower it to get more guessed joints outside the photo, raise it to remove stray shapes.
- `w * 0.25` in `drawStickman()`: thickness of the solids.
- `ortho()` in `draw()`: delete it for a perspective look.
- `modelType: "SINGLEPOSE_LIGHTNING"` in `loadPoseModel()`: faster, but only one person per photo.

## Notes and limits

- Photos come from Openverse and have different Creative Commons licenses, many non-commercial. Check each license before publishing. The API response includes attribution text if you want to credit the authors.
- The model sometimes misses people or invents joints, so some cells are empty and some shapes are distorted. That's part of the look.
- I wrote and checked the code, but I haven't confirmed in a browser that ml5 1.4.0 works with p5 2.3.4 in every setup. If detection doesn't start, check the console. The usual fix is loading p5 1.x instead.
