"""
Script 10: Graph-Theoretic Decomposition
Converts each sigil skeleton into a proper graph (nodes=junctions/endpoints,
edges=strokes between them), then computes graph metrics and spectral fingerprints.
"""
import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SIGIL_DIR = Path(r"C:\Users\PC\Downloads\goetia_analysis\extracted_sigils")
OUTDIR = Path(r"C:\Users\PC\Downloads\goetia_analysis")
GRAPH_DIR = OUTDIR / "graph_visualizations"
GRAPH_DIR.mkdir(exist_ok=True)

with open(OUTDIR / "sigil_metadata.json") as f:
    sigils = json.load(f)

DEMON_NAMES = {
    1:"Bael",2:"Agares",3:"Vassago",4:"Samigina",5:"Marbas",6:"Valefor",7:"Amon",
    8:"Barbatos",9:"Paimon",10:"Buer",11:"Gusion",12:"Sitri",13:"Beleth",14:"Leraje",
    15:"Eligos",16:"Zepar",17:"Botis",18:"Bathin",19:"Sallos",20:"Purson",21:"Marax",
    22:"Ipos",23:"Aim",24:"Naberius",25:"Glasya-Labolas",26:"Bune",27:"Ronove",
    28:"Berith",29:"Astaroth",30:"Forneus",31:"Foras",32:"Asmoday",33:"Gaap",
    34:"Furfur",35:"Marchosias",36:"Stolas",37:"Phenex",38:"Halphas",39:"Malphas",
    40:"Raum",41:"Focalor",42:"Vepar",43:"Sabnock",44:"Shax",45:"Vine",46:"Bifrons",
    47:"Uvall",48:"Haagenti",49:"Crocell",50:"Furcas",51:"Balam",52:"Alloces",
    53:"Camio",54:"Murmur",55:"Orobas",56:"Gremory",57:"Ose",58:"Amy",59:"Oriax",
    60:"Vapula",61:"Zagan",62:"Volac",63:"Andras",64:"Haures",65:"Andrealphus",
    66:"Cimejes",67:"Amdusias",68:"Belial",69:"Decarabia",70:"Seere",71:"Dantalion",
    72:"Andromalius"
}

def skeleton_to_graph(binary):
    """Convert a binary image to skeleton, then extract a graph."""
    skel = skeletonize(binary > 0).astype(np.uint8)

    # Compute neighbor count
    padded = np.pad(skel, 1, mode='constant')
    nc = np.zeros_like(skel, dtype=int)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            nc += padded[1+dy:padded.shape[0]-1+dy, 1+dx:padded.shape[1]-1+dx]

    skel_mask = skel > 0

    # Find junction pixels (>2 neighbors) and endpoints (1 neighbor)
    junction_mask = (nc > 2) & skel_mask
    endpoint_mask = (nc == 1) & skel_mask

    # Cluster junction pixels into junction nodes
    from scipy.ndimage import label as scipy_label
    junc_dilated = cv2.dilate(junction_mask.astype(np.uint8), np.ones((5,5), np.uint8))
    labeled, n_junctions = scipy_label(junc_dilated)

    G = nx.Graph()
    node_positions = {}
    node_id = 0

    # Add junction nodes (centroids of junction clusters)
    junction_node_ids = {}
    for j in range(1, n_junctions + 1):
        ys, xs = np.where(labeled == j)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        G.add_node(node_id, type='junction', pos=(cx, cy))
        node_positions[node_id] = (cx, cy)
        # Map all junction pixels in this cluster to this node
        for y, x in zip(ys, xs):
            if skel_mask[y, x] or junction_mask[y, x]:
                junction_node_ids[(y, x)] = node_id
        node_id += 1

    # Add endpoint nodes
    endpoint_pts = np.argwhere(endpoint_mask)
    endpoint_node_ids = {}
    for ey, ex in endpoint_pts:
        G.add_node(node_id, type='endpoint', pos=(int(ex), int(ey)))
        node_positions[node_id] = (int(ex), int(ey))
        endpoint_node_ids[(ey, ex)] = node_id
        node_id += 1

    # Trace edges along skeleton paths between nodes
    # For each skeleton pixel that's a "road" (exactly 2 neighbors), trace from it
    # Simple approach: for each endpoint/junction, do BFS along skeleton
    all_node_pixels = set(junction_node_ids.keys()) | set(endpoint_node_ids.keys())
    all_node_lookup = {**junction_node_ids, **endpoint_node_ids}

    visited_edges = set()

    def get_node_id(y, x):
        if (y, x) in endpoint_node_ids:
            return endpoint_node_ids[(y, x)]
        # Check if near a junction cluster
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if (y+dy, x+dx) in junction_node_ids:
                    return junction_node_ids[(y+dy, x+dx)]
        return None

    def trace_from(start_y, start_x, start_node):
        """Trace along skeleton from a node until hitting another node."""
        h, w = skel.shape
        edges_found = []

        # Get neighbors of start
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = start_y + dy, start_x + dx
                if 0 <= ny < h and 0 <= nx_ < w and skel[ny, nx_]:
                    neighbors.append((ny, nx_))

        for first_y, first_x in neighbors:
            edge_key = (min(start_node, -1), max(start_node, -1), first_y, first_x)

            # Trace along this path
            path_len = 1
            prev_y, prev_x = start_y, start_x
            cur_y, cur_x = first_y, first_x

            max_steps = 500
            steps = 0
            while steps < max_steps:
                steps += 1
                # Check if we've reached another node
                nid = get_node_id(cur_y, cur_x)
                if nid is not None and nid != start_node:
                    edge = tuple(sorted([start_node, nid]))
                    if edge not in visited_edges:
                        visited_edges.add(edge)
                        G.add_edge(start_node, nid, weight=path_len)
                        edges_found.append(edge)
                    break

                # Find next pixel (not the one we came from)
                found_next = False
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx_ = cur_y + dy, cur_x + dx
                        if (ny == prev_y and nx_ == prev_x):
                            continue
                        if 0 <= ny < h and 0 <= nx_ < w and skel[ny, nx_]:
                            prev_y, prev_x = cur_y, cur_x
                            cur_y, cur_x = ny, nx_
                            path_len += 1
                            found_next = True
                            break
                    if found_next:
                        break

                if not found_next:
                    break

        return edges_found

    # Trace from all endpoints
    for (ey, ex), nid in endpoint_node_ids.items():
        trace_from(ey, ex, nid)

    # Trace from all junctions
    for (jy, jx), nid in junction_node_ids.items():
        if skel[jy, jx]:
            trace_from(jy, jx, nid)

    return G, node_positions, skel

def graph_metrics(G):
    """Compute graph-theoretic metrics."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        return {"n_nodes":0,"n_edges":0,"avg_degree":0,"max_degree":0,
                "density":0,"n_components":0,"diameter":0,
                "avg_clustering":0,"avg_betweenness":0,"spectral_gap":0,
                "laplacian_eigenvalues":[]}

    degrees = [d for _, d in G.degree()]

    # Connected components
    components = list(nx.connected_components(G))
    n_components = len(components)

    # Diameter (of largest component)
    if n_nodes > 1:
        largest_cc = max(components, key=len)
        subG = G.subgraph(largest_cc)
        try:
            diameter = nx.diameter(subG)
        except:
            diameter = 0
    else:
        diameter = 0

    # Clustering coefficient
    avg_clustering = nx.average_clustering(G)

    # Betweenness centrality
    betweenness = nx.betweenness_centrality(G)
    avg_betweenness = np.mean(list(betweenness.values())) if betweenness else 0

    # Spectral properties (Laplacian eigenvalues)
    if n_nodes > 1:
        try:
            L = nx.laplacian_matrix(G).toarray().astype(float)
            eigenvalues = sorted(np.linalg.eigvalsh(L))
            spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 0
            # Keep first 10 eigenvalues as fingerprint
            eig_fingerprint = [round(float(e), 4) for e in eigenvalues[:10]]
        except:
            spectral_gap = 0
            eig_fingerprint = []
    else:
        spectral_gap = 0
        eig_fingerprint = []

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "avg_degree": round(np.mean(degrees), 3) if degrees else 0,
        "max_degree": max(degrees) if degrees else 0,
        "density": round(nx.density(G), 4),
        "n_components": n_components,
        "diameter": diameter,
        "avg_clustering": round(float(avg_clustering), 4),
        "avg_betweenness": round(float(avg_betweenness), 4),
        "spectral_gap": round(float(spectral_gap), 4),
        "laplacian_eigenvalues": eig_fingerprint,
    }


# Process all sigils
results = []
spectral_fingerprints = []
sigil_ids = []

for info in sigils[:72]:
    fpath = SIGIL_DIR / info["file"]
    if not fpath.exists():
        continue

    sid = info["id"]
    gray = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    G, positions, skel = skeleton_to_graph(binary)
    metrics = graph_metrics(G)

    result = {"id": sid, "name": DEMON_NAMES.get(sid, f"#{sid}"), **metrics}
    results.append(result)

    # Pad spectral fingerprint to length 10
    fp = metrics["laplacian_eigenvalues"]
    fp = fp + [0]*(10-len(fp))
    spectral_fingerprints.append(fp[:10])
    sigil_ids.append(sid)

    # Visualize graph for first few
    if sid <= 10 or sid in [43, 50, 72]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Original skeleton
        axes[0].imshow(skel, cmap='gray_r')
        axes[0].set_title(f"#{sid} {DEMON_NAMES.get(sid,'')} - Skeleton")
        axes[0].axis('off')

        # Graph overlay
        img_rgb = cv2.cvtColor(cv2.imread(str(fpath)), cv2.COLOR_BGR2RGB)
        axes[1].imshow(img_rgb)

        for node, data in G.nodes(data=True):
            if 'pos' in data:
                x, y = data['pos']
                color = 'red' if data.get('type') == 'junction' else 'lime'
                axes[1].plot(x, y, 'o', color=color, markersize=4, markeredgecolor='black', markeredgewidth=0.5)

        for u, v, data in G.edges(data=True):
            if 'pos' in G.nodes[u] and 'pos' in G.nodes[v]:
                x1, y1 = G.nodes[u]['pos']
                x2, y2 = G.nodes[v]['pos']
                axes[1].plot([x1, x2], [y1, y2], 'c-', linewidth=1, alpha=0.6)

        axes[1].set_title(f"Graph: {metrics['n_nodes']} nodes, {metrics['n_edges']} edges, D={metrics['diameter']}")
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(str(GRAPH_DIR / f"graph_{sid:03d}.png"), dpi=120)
        plt.close()

    if sid <= 5:
        print(f"Sigil {sid} ({DEMON_NAMES.get(sid,'')}): {metrics['n_nodes']} nodes, "
              f"{metrics['n_edges']} edges, diameter={metrics['diameter']}, "
              f"clustering={metrics['avg_clustering']:.3f}, spectral_gap={metrics['spectral_gap']:.3f}")

# Save results
with open(OUTDIR / "graph_analysis.json", "w") as f:
    json.dump(results, f, indent=2)

# Spectral similarity matrix
if spectral_fingerprints:
    fp_array = np.array(spectral_fingerprints)
    dist_matrix = squareform(pdist(fp_array, metric='euclidean'))

    # Find most spectrally similar pairs
    np.fill_diagonal(dist_matrix, np.inf)
    print(f"\n=== Top 10 Most Spectrally Similar Pairs ===")
    seen = set()
    flat_idx = np.argsort(dist_matrix.ravel())
    count = 0
    for idx in flat_idx:
        i, j = divmod(idx, len(sigil_ids))
        pair = (min(sigil_ids[i], sigil_ids[j]), max(sigil_ids[i], sigil_ids[j]))
        if pair not in seen:
            seen.add(pair)
            n1 = DEMON_NAMES.get(pair[0], f"#{pair[0]}")
            n2 = DEMON_NAMES.get(pair[1], f"#{pair[1]}")
            print(f"  #{pair[0]} {n1} <-> #{pair[1]} {n2}: spectral dist={dist_matrix[i,j]:.3f}")
            count += 1
            if count >= 10:
                break

    # Spectral distance heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    # Reset diagonal for display
    np.fill_diagonal(dist_matrix, 0)
    im = ax.imshow(dist_matrix, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(sigil_ids)))
    ax.set_yticks(range(len(sigil_ids)))
    ax.set_xticklabels([str(s) for s in sigil_ids], fontsize=5, rotation=90)
    ax.set_yticklabels([str(s) for s in sigil_ids], fontsize=5)
    ax.set_title("Spectral Distance Between Sigil Graphs")
    plt.colorbar(im, ax=ax, label="Euclidean distance (Laplacian eigenvalues)")
    plt.tight_layout()
    plt.savefig(str(OUTDIR / "spectral_distance_heatmap.png"), dpi=150)
    plt.close()

# Summary statistics
print(f"\n=== Graph Metrics Summary ({len(results)} sigils) ===")
for key in ["n_nodes", "n_edges", "avg_degree", "diameter", "avg_clustering", "spectral_gap"]:
    vals = [r[key] for r in results]
    print(f"  {key}: min={min(vals):.2f}, max={max(vals):.2f}, mean={np.mean(vals):.2f}")

# Degree distribution plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

nodes_list = [r["n_nodes"] for r in results]
edges_list = [r["n_edges"] for r in results]
diameters = [r["diameter"] for r in results]
clusterings = [r["avg_clustering"] for r in results]

axes[0].scatter(nodes_list, edges_list, alpha=0.6, s=40, c='steelblue')
axes[0].set_xlabel("Nodes"); axes[0].set_ylabel("Edges")
axes[0].set_title("Graph Size: Nodes vs Edges")

axes[1].hist(diameters, bins=15, color='coral', edgecolor='black')
axes[1].set_xlabel("Diameter"); axes[1].set_ylabel("Count")
axes[1].set_title("Graph Diameter Distribution")

axes[2].hist(clusterings, bins=15, color='mediumseagreen', edgecolor='black')
axes[2].set_xlabel("Avg Clustering Coefficient"); axes[2].set_ylabel("Count")
axes[2].set_title("Clustering Coefficient Distribution")

plt.tight_layout()
plt.savefig(str(OUTDIR / "graph_metrics_summary.png"), dpi=150)
plt.close()
print("\nSaved graph_analysis.json, spectral_distance_heatmap.png, graph_metrics_summary.png")
