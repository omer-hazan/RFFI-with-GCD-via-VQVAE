import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_original(orig_dist_dir, out_dir, use_mahalanobis):
    # load distances and mappings
    dists = np.load(os.path.join(orig_dist_dir, "distances_all.npy"))
    labels = np.load(os.path.join(orig_dist_dir, "labels_all.npy"))
    with open(os.path.join(orig_dist_dir, "code_to_class.json"), 'r') as f:
        code_to_class = {int(k.split()[1]): v for k, v in json.load(f).items()}

    n_samples, K = dists.shape
    num_classes = int(labels.max()) + 1

    # use Euclidean distance for prediction, even if mahalanobis is used for histograms
    euclid_dist_path = orig_dist_dir.replace("mahalanobis_", "")
    dists_for_pred = np.load(os.path.join(euclid_dist_path, "distances_all.npy"))
    nearest_codes = np.argmin(dists_for_pred, axis=1)
    predicted_classes = np.array([code_to_class[c] for c in nearest_codes])

    # ensure per-class output dirs
    for c in range(num_classes):
        os.makedirs(os.path.join(out_dir, "original", f"class_{c:02d}"), exist_ok=True)

    for c in range(num_classes):
        idxs = np.where(labels == c)[0]
        correct = idxs[predicted_classes[idxs] == c][:3]
        incorrect = idxs[predicted_classes[idxs] != c][:3]
        highlight_idxs = [cw for cw, cls in code_to_class.items() if cls == c]

        for phase, sel in [("correct", correct), ("incorrect", incorrect)]:
            for samp_idx in sel:
                inv_dist_vec = 1.0 / (dists[samp_idx] + 1e-8)

                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['#43A047' if i in highlight_idxs else 'gray' for i in range(K)]                
                ax.bar(np.arange(K), inv_dist_vec, color=colors, edgecolor='black', width=1.0)
                ax.set_xticks(highlight_idxs)
                ax.set_xticklabels(highlight_idxs, rotation=90, fontsize=6)
                ax.tick_params(axis='x', which='both', length=5)
                ax.set_title(f"Class {c:02d} | sample {samp_idx} ({phase})")
                ax.set_xlabel("Codeword index")
                ax.set_ylabel("1 / Mahalanobis Dist" if use_mahalanobis else "1 / Euclidean Dist")
                plt.tight_layout()

                fname = os.path.join(out_dir, "original", f"class_{c:02d}",
                                     f"samp{str(samp_idx).zfill(5)}_{phase}.png")
                fig.savefig(fname, dpi=150)
                plt.close(fig)

def plot_unseen(unseen_dist_dir, orig_dist_dir, out_dir, use_mahalanobis):
    dists_u = np.load(os.path.join(unseen_dist_dir, "unseen_distances.npy"))
    labels_u = np.load(os.path.join(unseen_dist_dir, "unseen_labels.npy"))
    K = dists_u.shape[1]

    dists_for_pred = np.load(os.path.join(orig_dist_dir.replace("mahalanobis_", ""), "distances_all.npy"))
    with open(os.path.join(orig_dist_dir, "code_to_class.json"), 'r') as f:
        code_to_class = {int(k.split()[1]): v for k, v in json.load(f).items()}

    unseen_classes = np.unique(labels_u)

    for c in unseen_classes:
        os.makedirs(os.path.join(out_dir, "unseen", f"class_{c:02d}"), exist_ok=True)

    for c in unseen_classes:
        idxs = np.where(labels_u == c)[0][:5]
        for samp_idx in idxs:
            inv_dist_vec = 1.0 / (dists_u[samp_idx] + 1e-8)
            fig, ax = plt.subplots(figsize=(8, 4))
            if 30 <= c <= 39:
                seen_class = c - 30
                highlight_idxs = [cw for cw, cls in code_to_class.items() if cls == seen_class]
                colors = ['#43A047' if i in highlight_idxs else 'gray' for i in range(K)]
            else:
                colors = ['gray'] * K
            
            ax.bar(np.arange(K), inv_dist_vec, color=colors, edgecolor='black', width=1.0)
            if 30 <= c <= 39:
                seen_class = c - 30
                highlight_idxs = [cw for cw, cls in code_to_class.items() if cls == seen_class]
                ax.set_xticks(highlight_idxs)
                ax.set_xticklabels(highlight_idxs, color='green', rotation=90, fontsize=6)

                # Predicting based on distances from seen classes
                nearest_code = np.argmin(dists_u[samp_idx])
                predicted_class = code_to_class[nearest_code]
                classified_correctly = (predicted_class == seen_class)
                title_classification = "Correct" if classified_correctly else "Incorrect"
                ax.set_title(f"Unseen class {c:02d} (seen as {seen_class:02d}) | sample {samp_idx} | {title_classification}")
            else:
                ax.set_xticks([])
                ax.set_title(f"Unseen class {c:02d} | sample {samp_idx}")

            ax.set_xlabel("Codeword index")
            ax.set_ylabel("1 / Mahalanobis Dist" if use_mahalanobis else "1 / Euclidean Dist")
            plt.tight_layout()

            fname = os.path.join(out_dir, "unseen", f"class_{c:02d}", f"samp{str(samp_idx).zfill(5)}.png")
            fig.savefig(fname, dpi=150)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate histograms of 1/distances for seen or unseen devices")
    parser.add_argument('--mode', choices=['original', 'unseen'], required=True,
                        help='"original" for seen devices, "unseen" for devices 30–44')
    parser.add_argument('--base_dir', default='analysis',
                        help='Base directory containing distances and unseen_distances')
    parser.add_argument('--mahalanobis', action='store_true',
                        help='Use Mahalanobis distance instead of Euclidean')
    args = parser.parse_args()

    orig_dist_dir = os.path.join(args.base_dir, 'mahalanobis_distances' if args.mahalanobis else 'distances_liv')
    unseen_dist_dir = os.path.join(args.base_dir, 'unseen_mahalanobis_distances' if args.mahalanobis else 'unseen_distances_liv')
    out_dir = os.path.join(args.base_dir, 'mahalanobis_histograms_per_class' if args.mahalanobis else 'euclidean_histograms_per_class_liv')

    for variant in ['original', 'unseen']:
        os.makedirs(os.path.join(out_dir, variant), exist_ok=True)

    if args.mode == 'original':
        plot_original(orig_dist_dir, out_dir, use_mahalanobis=args.mahalanobis)
    else:
        plot_unseen(unseen_dist_dir, orig_dist_dir, out_dir, use_mahalanobis=args.mahalanobis)

if __name__ == '__main__':
    main()
