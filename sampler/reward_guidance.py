from typing import Sequence
import math
import numpy as onp
import jax
import jax.numpy as jnp


class DAPRewardGuidance:
    def __init__(
        self,
        net,
        params,
        layers: Sequence[str],
        train_images,        # (B, C, H, W) - can be np or jnp
        train_labels,        # (B,)
        num_classes: int,    # e.g. 10 for CIFAR-10
        gamma: float = 1.0,
        batch_size: int = 128,  # for feature precomputation
    ):
        self.net = net
        self.params = params
        self.layers = list(layers)
        self.gamma = float(gamma)
        self.n_layers = len(self.layers)
        self.num_classes = int(num_classes)

        self._feat_batch_jit = jax.jit(self.features_flat_batch)

        # Store raw trainset if you want; it's not the memory hog.
        self.train_images = jnp.asarray(train_images)
        self.train_labels = jnp.asarray(train_labels)

        # default label for net.apply, if your model is conditional
        self.default_label = jnp.array(0, dtype=jnp.int32)

        # ---- NEW: precompute mean feature vector per label, streaming over batches ----
        self.label_feature_mean, self.label_counts = self._compute_label_feature_means(
            self.train_images, self.train_labels, batch_size=batch_size
        )
        # label_feature_mean shape: (num_classes, L)
        # label_counts shape:      (num_classes,)

    # ---------------- capture intermediates utils ----------------

    def _capture_pred(self, module, _):
        path = "/".join(module.scope.path).lower()
        return any(layer.lower() in path for layer in self.layers)

    def _extract_intermediates(self, x, label=None):
        if label is None:
            label = self.default_label

        _, state = self.net.apply(
            self.params,
            0.0,                    # s
            1.0,                    # t
            x,
            label,
            capture_intermediates=self._capture_pred,
            train=False,
            mutable=["intermediates"],
        )
        return state["intermediates"]

    def _collect_layer_acts_ordered(self, intermediates):
        unet_node = intermediates["flow_map"]["net"]["unet"]
        acts = []
        for layer in self.layers:
            parts = layer.split("/")
            node = unet_node
            for p in parts:
                node = node[p]
            act_container = node["__call__"]
            act = act_container[0][0]
            acts.append(act)
        return acts

    def _features_flat(self, x):
        intermediates = self._extract_intermediates(x)
        acts = self._collect_layer_acts_ordered(intermediates)
        flat_parts = [jnp.ravel(jnp.asarray(a)) for a in acts]
        if len(flat_parts) == 0:
            return jnp.zeros((0,), dtype=jnp.float32)
        return jnp.concatenate(flat_parts, axis=0)  # (L,)

    def features_flat_batch(self, x_batch):
        # x_batch: (B, C, H, W) -> (B, L)
        return jax.vmap(self._features_flat)(x_batch)

    # ---------------- NEW: streaming computation of label-wise means ----------------

    def _compute_label_feature_means(self, images, labels, batch_size: int):
        """
        images: (B, C, H, W), labels: (B,)
        Returns:
          label_feature_mean: (num_classes, L)
          label_counts:       (num_classes,)
        We *don't* build a (B, L) array; we stream in small batches.
        """
        images = onp.asarray(images)   # host arrays for Python looping
        labels = onp.asarray(labels)

        B = images.shape[0]

        label_sums = None   # will be (num_classes, L) once L known
        label_counts = onp.zeros(self.num_classes, dtype=onp.int64)

        # process in small batches to limit activation + feature memory
        for start in range(0, B, batch_size):
            end = min(start + batch_size, B)
            x_chunk = jnp.asarray(images[start:end])   # (bs, C, H, W)
            y_chunk = labels[start:end]               # (bs,)

            # compute features for this chunk on device
            feats_chunk = self._feat_batch_jit(x_chunk) # (bs, L)
            feats_np = onp.asarray(feats_chunk)              # back to host for simple accumulation

            if label_sums is None:
                L = feats_np.shape[1]
                label_sums = onp.zeros((self.num_classes, L), dtype=feats_np.dtype)

            # accumulate per-label sums and counts
            for c in range(self.num_classes):
                mask = (y_chunk == c)
                if mask.any():
                    label_sums[c] += feats_np[mask].sum(axis=0)
                    label_counts[c] += mask.sum()

        # convert sums to means
        label_feature_mean = onp.zeros_like(label_sums)
        for c in range(self.num_classes):
            if label_counts[c] > 0:
                label_feature_mean[c] = label_sums[c] / float(label_counts[c])

        # back to jax arrays
        return (
            jnp.asarray(label_feature_mean, dtype=jnp.float32),
            jnp.asarray(label_counts),
        )

    # ---------------- avg_grad using per-label mean features ----------------

    def avg_grad(self, x_synth, label):
        """
        x_synth: (C, H, W)
        label: scalar int (e.g. 0..9)
        Uses ONLY the per-label mean feature vector, not all training features.
        """
        label = int(label)
        # VJP once for x_synth
        _, pullback = jax.vjp(lambda x: self._features_flat(x), x_synth)

        # average-over-layers factor
        scale = 1.0 / max(1, self.n_layers)

        # precomputed mean feature vector for this label: (L,)
        mean_feat = self.label_feature_mean[label]  # (L,)

        v = scale * mean_feat  # cotangent for VJP
        grad_x, = pullback(v)  # shape (C, H, W)

        avg_grad = grad_x * -self.gamma
        return avg_grad

