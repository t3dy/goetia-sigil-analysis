# Goetic Sigil Analysis: Reverse-Engineering Demonic Circuit Diagrams

A computational image analysis pipeline that treats the 72 seals/sigils from the *Goetia of Dr. Rudd* as visual data, using computer vision and topology to reverse-engineer their construction grammar.

The sigils from the Lesser Key of Solomon have a striking resemblance to circuit diagrams or node-and-wire schematics. This project asks: **can we decompose them into a formal visual vocabulary?** What primitives were they built from? Do they cluster into structural families?

![Segmentation](segmentation_vis.png)

## The Pipeline

### Script 1: Grid Segmentation (`01_segment_sigils.py`)

**Problem:** The source image contains all 72 sigils in a dense grid with text labels. We need to isolate each one cleanly.

**Approach:** Rather than trying to detect the grid lines directly (which are implicit whitespace, not drawn lines), this script uses **morphological dilation** on the binarized ink to "grow" nearby strokes into connected blobs. Each blob becomes one sigil's bounding box.

The key design choice is automatically tuning the dilation iterations. Too little dilation (1 iteration) and a single sigil splinters into many disconnected stroke fragments. Too much (5+ iterations) and adjacent sigils merge into mega-blobs. The script sweeps dilation levels 1-7 and picks whichever yields a count closest to ~76 regions (72 sigils plus a few variant labels like "9b. Paimon").

After detection, bounding boxes are sorted into reading order by clustering their y-coordinates into rows (using a 40px gap threshold), then sorting each row left-to-right by x.

**Output:** 78 individual sigil images + `sigil_metadata.json` with bounding box coordinates and aspect ratios.

---

### Script 2: Skeleton & Topology (`02_skeleton_analysis.py`)

**Problem:** The raw sigil strokes have varying thickness, decorative fills, and visual noise. We want the pure topological structure — the abstract "wiring diagram."

**Approach:** [Zhang-Suen skeletonization](https://en.wikipedia.org/wiki/Zhang%E2%80%93Suen_thinning_algorithm) via scikit-image reduces every stroke to a 1-pixel-wide skeleton while preserving connectivity. This is the morphological equivalent of tracing the centerline of every wire in a circuit.

From the skeleton we extract five topological invariants:
- **Connected components** — how many separate pieces make up the sigil (mean: 18.2, range: 3-47)
- **Junctions** — skeleton pixels with >2 neighbors, i.e., branch points where the network forks
- **Endpoints** — skeleton pixels with exactly 1 neighbor, i.e., terminal tips
- **Holes** — enclosed regions (computed via contour hierarchy difference between RETR_TREE and RETR_EXTERNAL)
- **Euler number** — components minus holes, a topological invariant that characterizes the "genus" of the design

The junction-to-endpoint ratio is particularly revealing: high values mean a dense branching network (circuit-like), low values mean mostly free-floating strokes.

**Key finding:** Most sigils are assemblies of 10-20 disconnected components, not single continuous strokes. They're modular constructions.

---

### Script 3: Junction & Endpoint Detection (`03_junction_endpoint_detection.py`)

**Problem:** Script 2 counts junctions and endpoints; this script localizes them spatially and classifies the terminal decorations.

**Approach:** After skeletonization, we compute an 8-connected neighbor count for every skeleton pixel. Pixels with >2 neighbors are junctions (branch points); pixels with exactly 1 neighbor are endpoints (terminals).

Raw junction pixels tend to cluster in small groups at each actual branch point (because a 3-way intersection occupies several pixels in the skeleton). We use `scipy.ndimage.label` on a dilated junction mask to merge these into distinct junction clusters.

For endpoint classification, we examine a 15-pixel radius around each terminal and look for:
- **Circles** — contours with circularity > 0.7 (the classic "bubble" terminals)
- **Filled/cross** — regions with >30% ink density (solid decorative endings)
- **Horizontal/vertical bars** — elongated ink patches perpendicular to the stroke
- **Simple** — bare line endings with no decoration

**Key finding:** The terminal vocabulary is small (~5 types), with simple bare endings dominating (2,868), followed by filled/cross decorations (480) and circles (133). This suggests a constrained decorative grammar.

---

### Script 4: Hough Line & Circle Detection (`04_hough_geometry.py`)

**Problem:** Are these sigils built on a regular angular grid, or are the line orientations freeform?

**Approach:** The [Hough transform](https://en.wikipedia.org/wiki/Hough_transform) is a classical technique for detecting parametric shapes (lines, circles) in edge images. We apply:

- **Probabilistic Hough Lines** (`HoughLinesP`) with minimum length 15px and maximum gap 5px to detect straight segments and measure their angles
- **Hough Circles** (`HoughCircles`) with radius range 3-50px to detect circular elements (terminal bubbles, loops, arcs)

For each sigil we build a 12-bin angle histogram (0°-180° in 15° bins) and identify the dominant orientation. The per-sigil histograms are also aggregated into a global orientation distribution.

**Key finding:** The global angle distribution shows a **massive bias toward 0° (horizontal) and 90° (vertical)**, confirming these are overwhelmingly built on an orthogonal grid. The "circuit diagram" intuition is geometrically validated. Diagonal elements exist but are a distinct minority. The lines-vs-circles scatter plot also reveals that most sigils have 15-50 line segments and 10-30 circular elements, with a roughly linear relationship.

---

### Script 5: Feature Extraction (`05_feature_extraction.py`)

**Problem:** We need a unified, comparable representation of each sigil for clustering and similarity analysis.

**Approach:** This script computes a comprehensive feature vector per sigil combining several analysis modalities:

- **Ink ratio** — what fraction of the bounding box contains ink (overall density)
- **Aspect ratio** — width/height, distinguishing wide vs tall sigils
- **Compactness** — ratio of ink pixels to the ink bounding box area (how tightly packed the strokes are)
- **Box-counting fractal dimension** — covers the image with boxes of size 2, 4, 8, 16, 32, 64 and counts how many contain ink, then fits a log-log regression. Values between 1.0 (line) and 2.0 (filled area) characterize the structural complexity. Most sigils land at 1.2-1.5, consistent with branching tree/network structures.
- **Bilateral symmetry** — Pearson correlation between the image and its horizontal/vertical mirror. Most sigils are asymmetric (scores ~0.2), but a few (Oriax #59, Marbas #5) show strong symmetry.
- **Radial profile** — divides the image into 8 concentric annular bins from center to edge and measures ink density in each. Reveals that these are centripetally organized: dense center, sparse edges.
- **Quadrant density** — ink coverage in each image quadrant, capturing spatial balance.

**Design rationale:** Each feature captures a different geometric "view" of the sigil. Together they create a multi-dimensional fingerprint that two sigils will share only if they're structurally similar across all these dimensions simultaneously.

---

### Script 6: Clustering & Similarity (`06_clustering.py`)

**Problem:** Do the 72 sigils fall into natural structural families? Which ones are most alike?

**Approach:** This script combines all features from scripts 2-5 into a single 38-dimensional feature matrix, then applies:

1. **StandardScaler normalization** — essential because features have wildly different scales (fractal dimension ~1.3 vs skeleton length ~300px)
2. **Ward hierarchical clustering** — builds a complete merge tree (dendrogram) using Ward's minimum variance criterion, which tends to produce compact, equal-sized clusters
3. **PCA projection** — reduces 38 dimensions to 2 for visualization, showing how sigils spread in the most informative plane (PC1 captures 28.7% of variance, PC2 captures 13.7%)
4. **Pairwise distance ranking** — finds the most structurally similar sigil pairs across all features

We cut the dendrogram at 8 clusters and characterize each by its most distinctive features (largest z-score deviations from the global mean).

**Key finding:** The 8 families have interpretable structural signatures:
- A large family of sparse, spread-out designs (22 sigils)
- A family of unified, vertically-oriented designs (15 sigils)
- A small family of loop-heavy, circuit-like designs (6 sigils, the most "electrical")
- An outlier (Zepar #16) that's structurally unique

The most similar pairs (Amon & Osé, Furfur & Halphas) could indicate shared construction templates or copying between entries in the grimoire tradition.

---

## Results At A Glance

### Construction Grammar Hypothesis

From the combined analysis, these sigils appear to be built from:

1. **A horizontal/vertical backbone** — the dominant orthogonal grid confirmed by Hough analysis
2. **Branch points** (junction nodes) where the network forks — mean 54 per sigil
3. **A vocabulary of ~5 terminal decorations** — circles, crosses/fills, horizontal bars, vertical bars, bare endpoints
4. **Modular assembly** — mean 18 disconnected components per sigil, not continuous strokes
5. **Centripetal organization** — dense center radiating outward
6. **Rare enclosed loops** — only ~30% of sigils contain holes

### Feature Distributions

![Features](feature_distributions.png)

### Line Orientation Analysis

![Angles](angle_distribution.png)

### Dendrogram (8 Families)

![Dendrogram](dendrogram.png)

### PCA Cluster Map

![PCA](pca_clusters.png)

## Requirements

```
opencv-python-headless
numpy
scipy
scikit-image
scikit-learn
matplotlib
networkx
```

## Usage

Place the source image as `../1280px-72_Goeta_sigils.png` (or edit the `INPUT` path in script 1), then run sequentially:

```bash
python 01_segment_sigils.py
python 02_skeleton_analysis.py
python 03_junction_endpoint_detection.py
python 04_hough_geometry.py
python 05_feature_extraction.py
python 06_clustering.py
```

Each script produces JSON data files and PNG visualizations in subdirectories.

## Future Directions

- **Graph-theoretic comparison** — convert each skeleton to a proper graph (nodes=junctions, edges=strokes) and compare using graph edit distance or spectral methods
- **Generative model** — learn a probabilistic grammar that can synthesize new "valid" sigils
- **Historical correlation** — compare sigil complexity features against the ordering in different grimoire manuscripts to detect editorial patterns
- **Fourier descriptors** — encode contour shapes for rotation-invariant similarity
- **Template matching** — detect recurring sub-motifs shared across multiple sigils
