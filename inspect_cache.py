"""Inspect a scene_info cache pickle produced by RGBDDataset.__init__
(droid_slam/data_readers/base.py), e.g. droid_slam/data_readers/cache/vslamlab.pickle

Usage:
    python inspect_cache.py droid_slam/data_readers/cache/vslamlab.pickle
    python inspect_cache.py droid_slam/data_readers/cache/vslamlab.pickle --scene <scene_name>

    # inspect the co-visibility graph for one scene (optionally one frame)
    python inspect_cache.py cache.pickle --scene <scene_name> --graph
    python inspect_cache.py cache.pickle --scene <scene_name> --graph --frame 42

    # render a grid of sample rgb thumbnails for a scene
    python inspect_cache.py cache.pickle --scene <scene_name> --thumbnails

    # training-relevant stats over the whole cache (size, variety, dispersion)
    python inspect_cache.py cache.pickle --stats
    python inspect_cache.py cache.pickle --stats --n-frames 2 --fmin 8 --fmax 75 --hist

Note: --stats treats every scene in the pickle as training data. It does not
apply a dataset's train/test split (e.g. VSLAMLAB.is_test_scene), since that
logic lives in the dataset class, not the cache file.
"""

import argparse
import pickle

import numpy as np


def summarize_scene(name, info):
    n_images = len(info['images'])
    n_depths = len(info['depths'])
    n_graph = len(info['graph'])

    print(f"\nScene: {name}")
    print(f"  images:     {n_images} (first: {info['images'][0]})")
    print(f"  depths:     {n_depths} (first: {info['depths'][0]})")
    print(f"  poses:      shape {np.asarray(info['poses']).shape}")
    print(f"  intrinsics: {info['intrinsics'][0]}")
    print(f"  graph:      {n_graph} frames indexed")

    degrees = [len(j) for j, _ in info['graph'].values()]
    if degrees:
        print(f"  graph co-visible frame count: min={min(degrees)}, "
              f"max={max(degrees)}, mean={np.mean(degrees):.1f}")


def show_graph(name, info, frame=None):
    graph = info['graph']

    if frame is not None:
        if frame not in graph:
            print(f"Frame {frame} not present in graph for scene '{name}'. "
                  f"Valid indices: 0..{max(graph.keys())}")
            return
        j, d = graph[frame]
        order = np.argsort(d)
        print(f"\nScene: {name}, frame {frame}: {len(j)} co-visible frames")
        for jj, dd in zip(j[order], d[order]):
            print(f"  frame {jj:5d}  flow_dist={dd:.2f}")
        return

    import matplotlib.pyplot as plt

    n = len(info['images'])
    matrix = np.full((n, n), np.nan, dtype=np.float32)
    for i, (j, d) in graph.items():
        matrix[i, j] = d

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar(label='flow distance')
    plt.title(f"Co-visibility graph: {name}")
    plt.xlabel('frame j')
    plt.ylabel('frame i')
    plt.tight_layout()
    out_path = f"{name.replace('/', '_')}_graph.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved graph heatmap to '{out_path}'")


def show_thumbnails(name, info, n=16, cols=4):
    import cv2
    import matplotlib.pyplot as plt

    images = info['images']
    idxs = np.linspace(0, len(images) - 1, min(n, len(images)), dtype=int)

    rows = (len(idxs) + cols - 1) // cols
    _, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, idx in zip(axes, idxs):
        img = cv2.imread(str(images[idx]))
        if img is not None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"frame {idx}", fontsize=9)
        ax.axis('off')

    for ax in axes[len(idxs):]:
        ax.axis('off')

    plt.suptitle(f"Thumbnails: {name}")
    plt.tight_layout()
    out_path = f"{name.replace('/', '_')}_thumbnails.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved thumbnails to '{out_path}'")


def scene_training_stats(name, info, n_frames, fmin, fmax):
    """ per-scene metrics that mirror how RGBDDataset actually consumes the graph:
    - eligible: frames usable as a training anchor (base.py _build_dataset_index: degree > n_frames)
    - degenerate: anchors with zero co-visible frames inside (fmin, fmax) -- __getitem__ will
      then repeat the same frame instead of sampling a real second view (base.py __getitem__)
    """
    graph = info['graph']
    n = len(info['images'])

    degrees = np.array([len(j) for j, _ in graph.values()])
    eligible = int(np.sum(degrees > n_frames))

    usable_degrees = np.array([
        int(np.sum((d > fmin) & (d < fmax))) for _, d in graph.values()
    ])
    degenerate = int(np.sum(usable_degrees == 0))

    all_d = np.concatenate([d for _, d in graph.values()]) if len(graph) else np.array([])

    poses = np.asarray(info['poses'])
    xyz = poses[:, :3]
    bbox_diag = float(np.linalg.norm(xyz.max(0) - xyz.min(0))) if len(xyz) else 0.0
    path_length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()) if len(xyz) > 1 else 0.0

    return {
        'name': name,
        'n_frames': n,
        'eligible': eligible,
        'degenerate': degenerate,
        'mean_degree': float(degrees.mean()) if len(degrees) else 0.0,
        'flow_dist': all_d,
        'bbox_diag': bbox_diag,
        'path_length': path_length,
    }


def training_stats(scene_info, n_frames=2, fmin=8.0, fmax=75.0, top_k=None, hist=False):
    rows = [scene_training_stats(name, info, n_frames, fmin, fmax)
            for name, info in scene_info.items()]

    total_frames = sum(r['n_frames'] for r in rows)
    total_eligible = sum(r['eligible'] for r in rows)
    total_degenerate = sum(r['degenerate'] for r in rows)
    all_flow = np.concatenate([r['flow_dist'] for r in rows if len(r['flow_dist'])])
    bbox_diags = np.array([r['bbox_diag'] for r in rows])
    path_lengths = np.array([r['path_length'] for r in rows])

    print(f"\n=== Training stats (n_frames={n_frames}, fmin={fmin}, fmax={fmax}) ===")
    print(f"Scenes:                {len(rows)}")
    print(f"Total frames:          {total_frames}")
    print(f"Eligible anchors:      {total_eligible}  "
          f"(~ size of dataset_index, i.e. training samples per epoch)")
    print(f"Degenerate anchors:    {total_degenerate} "
          f"({100 * total_degenerate / max(total_frames, 1):.1f}%) "
          f"-- no co-visible frame inside (fmin, fmax), __getitem__ will duplicate frames")

    if len(all_flow):
        print(f"\nEdge flow-distance (variety of motion between co-visible frames):")
        print(f"  min={all_flow.min():.1f}  mean={all_flow.mean():.1f}  "
              f"median={np.median(all_flow):.1f}  max={all_flow.max():.1f}  std={all_flow.std():.1f}")

    print(f"\nScene translation dispersion (bbox diagonal, scaled units):")
    print(f"  min={bbox_diags.min():.2f}  mean={bbox_diags.mean():.2f}  max={bbox_diags.max():.2f}")
    print(f"Scene path length (trajectory length, scaled units):")
    print(f"  min={path_lengths.min():.2f}  mean={path_lengths.mean():.2f}  max={path_lengths.max():.2f}")

    ranked = sorted(rows, key=lambda r: r['eligible'])
    shown = ranked if top_k is None else ranked[:top_k]
    print(f"\nWeakest scenes by eligible-anchor count"
          f"{'' if top_k is None else f' (worst {top_k})'}:")
    print(f"  {'scene':<40} {'frames':>7} {'eligible':>9} {'degenerate':>10} {'mean_deg':>9}")
    for r in shown:
        print(f"  {r['name']:<40} {r['n_frames']:>7} {r['eligible']:>9} "
              f"{r['degenerate']:>10} {r['mean_degree']:>9.1f}")

    if hist:
        import matplotlib.pyplot as plt

        _, axes = plt.subplots(1, 2, figsize=(11, 4))
        if len(all_flow):
            axes[0].hist(all_flow, bins=50)
        axes[0].set_title('Edge flow-distance distribution')
        axes[0].set_xlabel('flow distance')

        eligible_counts = [r['eligible'] for r in rows]
        axes[1].hist(eligible_counts, bins=30)
        axes[1].set_title('Eligible-anchor count per scene')
        axes[1].set_xlabel('eligible anchors')

        plt.tight_layout()
        out_path = 'training_stats_hist.png'
        plt.savefig(out_path, dpi=150)
        print(f"\nSaved histograms to '{out_path}'")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pickle_path', help="path to the cached scene_info .pickle file")
    parser.add_argument('--scene', default=None, help="only show this scene (full detail)")
    parser.add_argument('--graph', action='store_true',
                         help="inspect the co-visibility graph for --scene "
                              "(heatmap, or edges for --frame if given)")
    parser.add_argument('--frame', type=int, default=None,
                         help="with --graph, show co-visible frames/distances for this frame index")
    parser.add_argument('--thumbnails', action='store_true',
                         help="save a grid of sample rgb thumbnails for --scene")
    parser.add_argument('--n-thumbnails', type=int, default=16,
                         help="number of thumbnails to sample (default: 16)")
    parser.add_argument('--stats', action='store_true',
                         help="show training-relevant size/variety/dispersion metrics "
                              "over all scenes in the cache")
    parser.add_argument('--n-frames', type=int, default=2,
                         help="n_frames used by RGBDDataset (default: 2, matches VSLAMLAB)")
    parser.add_argument('--fmin', type=float, default=8.0, help="fmin used by RGBDDataset (default: 8.0)")
    parser.add_argument('--fmax', type=float, default=75.0, help="fmax used by RGBDDataset (default: 75.0)")
    parser.add_argument('--top-k', type=int, default=None,
                         help="with --stats, only list the K weakest scenes (default: all)")
    parser.add_argument('--hist', action='store_true',
                         help="with --stats, save histograms of edge flow-distance and "
                              "per-scene eligible-anchor counts")
    args = parser.parse_args()

    with open(args.pickle_path, 'rb') as f:
        scene_info = pickle.load(f)[0]

    print(f"Loaded '{args.pickle_path}': {len(scene_info)} scenes")

    if args.stats:
        training_stats(scene_info, n_frames=args.n_frames, fmin=args.fmin, fmax=args.fmax,
                        top_k=args.top_k, hist=args.hist)
        return

    if args.scene is None:
        if args.graph or args.thumbnails:
            parser.error("--graph/--thumbnails require --scene")
        for name, info in scene_info.items():
            summarize_scene(name, info)
        return

    if args.scene not in scene_info:
        print(f"Scene '{args.scene}' not found. Available scenes:")
        for name in scene_info:
            print(f"  {name}")
        return

    info = scene_info[args.scene]
    summarize_scene(args.scene, info)

    if args.graph:
        show_graph(args.scene, info, frame=args.frame)

    if args.thumbnails:
        show_thumbnails(args.scene, info, n=args.n_thumbnails)


if __name__ == '__main__':
    main()
