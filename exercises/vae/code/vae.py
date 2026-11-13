"""Activity 4 - Variational Autoencoder on MNIST (Exercises 1 and 2).

Run from the folder that should hold the MNIST download (``data/``), e.g.

    cd docs/exercises/vae && python code/vae.py

Figures and every number quoted in the report go to ``figures/`` (next to ``code/``);
the numbers are in ``figures/results.json``.

Order of work: both models are trained first, using only the train and validation
splits. The test set is touched only after training is over, for the final numbers,
the figures and the diagnostics quoted in the analysis.
"""
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import random_split
from torchvision import datasets, transforms

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)          # fixed thread count -> identical floating-point results run to run
T0 = time.time()

# --8<-- [start:data]
# ---------------------------------------------------------------- 1A: data
to_tensor = transforms.ToTensor()                      # uint8 [0, 255] -> float [0, 1]
train_full = datasets.MNIST("data", train=True, download=True, transform=to_tensor)
test_set = datasets.MNIST("data", train=False, download=True, transform=to_tensor)
train_set, val_set = random_split(train_full, [50_000, 10_000],
                                  generator=torch.Generator().manual_seed(42))


def stack(ds):
    """Whole split as an (N, 784) float tensor in [0, 1] and its labels.
    Goes through the dataset's ToTensor transform, then flattens 28x28 -> 784."""
    xs, ys = zip(*(ds[i] for i in range(len(ds))))
    return torch.stack(xs).reshape(len(ds), 784), torch.tensor(ys)


Xtr, ytr = stack(train_set)
Xva, yva = stack(val_set)
Xte, yte = stack(test_set)          # NOT used until both models are trained
# --8<-- [end:data]
print("shapes", tuple(Xtr.shape), tuple(Xva.shape), tuple(Xte.shape),
      "range", Xtr.min().item(), Xtr.max().item(), flush=True)


# --8<-- [start:model]
# ---------------------------------------------------------------- 1B: model
class VAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        # encoder 784 -> 512 -> 256 (ReLU), then two linear heads 256 -> d
        self.enc = nn.Sequential(nn.Linear(784, 512), nn.ReLU(),
                                 nn.Linear(512, 256), nn.ReLU())
        self.mu_head = nn.Linear(256, d)
        self.logvar_head = nn.Linear(256, d)
        # decoder d -> 256 -> 512 -> 784 (ReLU on hidden layers, sigmoid on the output)
        self.dec = nn.Sequential(nn.Linear(d, 256), nn.ReLU(),
                                 nn.Linear(256, 512), nn.ReLU(),
                                 nn.Linear(512, 784))   # pre-sigmoid values (logits)

    def encode(self, x):
        h = self.enc(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar, generator=None):
        # z = mu + sigma * eps, eps ~ N(0, I): the randomness sits in eps, so the
        # gradient flows through mu and sigma
        eps = torch.randn(mu.shape, generator=generator)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode_logits(self, z):
        return self.dec(z)

    def decode(self, z):
        # sigmoid output = Bernoulli mean of each pixel (what the figures show)
        return torch.sigmoid(self.dec(z))

    def forward(self, x, generator=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, generator)
        return self.decode_logits(z), mu, logvar


def loss_terms(logits, x, mu, logvar):
    """Per-image recon and KL (nats), shape (B,). Both written by hand.

    recon: Bernoulli negative log-likelihood (BCE) SUMMED over the 784 pixels.
        With p = sigmoid(l):  -[x log p + (1 - x) log(1 - p)] = softplus(l) - x * l,
        the same quantity written so that log(0) never appears when the sigmoid saturates.
    KL(N(mu, sigma^2) || N(0, I)) in closed form, summed over the d latent dimensions:
        -1/2 * sum(1 + log sigma^2 - mu^2 - sigma^2)
    """
    recon = (F.softplus(logits) - x * logits).sum(dim=1)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    return recon, kl


def vae_loss(logits, x, mu, logvar, beta=1.0):
    """Minimised objective: (sum of recon + beta * sum of KL) / batch size."""
    recon, kl = loss_terms(logits, x, mu, logvar)
    return (recon.sum() + beta * kl.sum()) / x.shape[0], recon.sum().item(), kl.sum().item()
# --8<-- [end:model]


# --8<-- [start:train]
# ---------------------------------------------------------------- 1C: training
@torch.no_grad()
def evaluate(model, X, bs=2000, seed=123):
    """Mean recon and KL per image (nats) on X, one sampled z per image.
    A private generator (fixed seed) keeps the number reproducible and leaves the
    global RNG used by training untouched."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    r = k = 0.0
    for i in range(0, len(X), bs):
        x = X[i:i + bs]
        logits, mu, logvar = model(x, g)
        rec, kl = loss_terms(logits, x, mu, logvar)
        r += rec.sum().item()
        k += kl.sum().item()
    return r / len(X), k / len(X)


def fit(d, beta=1.0, epochs=20, lr=1e-3, bs=128):
    torch.manual_seed(42)                                   # same init for every model
    model = VAE(d)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    order = torch.Generator().manual_seed(42)               # mini-batch order
    hist = {k: [] for k in ["train_recon", "train_kl", "val_recon", "val_kl"]}
    first_epoch_steps = []                                  # per-batch terms, epoch 1 only
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=order)
        sr = sk = 0.0
        for step, i in enumerate(range(0, len(Xtr), bs)):
            x = Xtr[perm[i:i + bs]]
            logits, mu, logvar = model(x)
            loss, r, k = vae_loss(logits, x, mu, logvar, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sr += r
            sk += k
            if ep == 1:
                first_epoch_steps.append([step, r / len(x), k / len(x)])
        # train point = average over the epoch's mini-batches (the model moves during it);
        # validation point = measured with the weights at the end of the epoch
        va_r, va_k = evaluate(model, Xva)
        hist["train_recon"].append(sr / len(Xtr))
        hist["train_kl"].append(sk / len(Xtr))
        hist["val_recon"].append(va_r)
        hist["val_kl"].append(va_k)
        print(f"d={d:2d} epoch {ep:2d} | train recon {sr/len(Xtr):7.2f} KL {sk/len(Xtr):6.2f}"
              f" | val recon {va_r:7.2f} KL {va_k:6.2f}", flush=True)
    return model.eval(), hist, first_epoch_steps
# --8<-- [end:train]


def show(ax, img):
    ax.imshow(img.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])


def tile(imgs, nrow, ncol):
    """(nrow*ncol, 784) -> one (nrow*28, ncol*28) image, row-major."""
    return imgs.reshape(nrow, ncol, 28, 28).permute(0, 2, 1, 3).reshape(nrow * 28, ncol * 28).numpy()


res = {"env": dict(python=platform.python_version(), torch=torch.__version__,
                   threads=torch.get_num_threads())}

# --8<-- [start:check]
# sanity check: the hand-written BCE equals torch's F.binary_cross_entropy(reduction="sum")
# (float64, so that sigmoid(l) never rounds to exactly 0 or 1, where torch clamps log at -100)
with torch.no_grad():
    l = torch.randn(256, 784, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 4
    x = Xva[:256].double()
    mine = (F.softplus(l) - x * l).sum()
    ref = F.binary_cross_entropy(torch.sigmoid(l), x, reduction="sum")
res["bce_check"] = dict(hand_written=mine.item(), torch_bce=ref.item(),
                        rel_diff=abs(mine.item() - ref.item()) / ref.item())
# --8<-- [end:check]
print("BCE check", res["bce_check"], flush=True)

# --8<-- [start:fit]
# ============ train both models first (train + validation splits only)
m16, h16, steps16 = fit(16)          # Exercise 1
m2, h2, _ = fit(2)                   # Exercise 2: identical except d = 2
# ============ test set used from here on, once, for the final numbers
te16 = evaluate(m16, Xte)
te2 = evaluate(m2, Xte)
# --8<-- [end:fit]

S = np.array(steps16)                     # columns: step, recon, KL (per image, one batch)
res["ex1"] = dict(test_recon=te16[0], test_kl=te16[1], test_neg_elbo=sum(te16),
                  val_kl_epoch1=h16["val_kl"][0], val_kl_epoch20=h16["val_kl"][-1],
                  history=h16, epoch1_steps=steps16,
                  epoch1_summary=dict(
                      n_steps=len(S),
                      kl_step0=S[0, 2], kl_step20=S[20, 2], kl_last_step=S[-1, 2],
                      total_step0=S[0, 1] + S[0, 2], total_step20=S[20, 1] + S[20, 2],
                      total_last_step=S[-1, 1] + S[-1, 2],
                      transient_kl_max=S[:20, 2].max(), transient_kl_argmax=int(S[:20, 2].argmax()),
                      kl_min_after_transient=S[10:, 2].min(),
                      kl_argmin_after_transient=int(10 + S[10:, 2].argmin()),
                      recon_last_step=S[-1, 1]))
res["ex2"] = dict(test_recon=te2[0], test_kl=te2[1], test_neg_elbo=sum(te2),
                  val_kl_epoch1=h2["val_kl"][0], val_kl_epoch20=h2["val_kl"][-1], history=h2)
print("TEST d=16 recon %.2f KL %.2f -ELBO %.2f" % (*te16, sum(te16)))
print("TEST d=2  recon %.2f KL %.2f -ELBO %.2f" % (*te2, sum(te2)), flush=True)

# --8<-- [start:fig1]
# ---------------------------------------------------------------- Figure 1
ep = np.arange(1, 21)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, key, name in [(ax[0], "recon", "Reconstruction term (BCE)"), (ax[1], "kl", "KL term")]:
    a.plot(ep, h16[f"train_{key}"], "o-", ms=3, label="train (epoch average)")
    a.plot(ep, h16[f"val_{key}"], "s-", ms=3, label="validation (end of epoch)")
    a.set_title(f"{name} per epoch, d = 16")
    a.set_xlabel("epoch")
    a.set_ylabel("nats per image")
    a.set_xticks([1, 5, 10, 15, 20])
    a.grid(alpha=.3)
    a.legend()
fig.suptitle("Figure 1 - VAE training curves on MNIST (d = 16, beta = 1)")
fig.tight_layout()
fig.savefig(OUT / "fig1-training-curves.png", dpi=150)
plt.close(fig)
# --8<-- [end:fig1]

# --8<-- [start:fig2]
# ---------------------------------------------------------------- Figure 2
first_idx = [int((yte == c).nonzero()[0]) for c in range(10)]   # first test image of each digit
with torch.no_grad():
    mu, _ = m16.encode(Xte[first_idx])
    rec = m16.decode(mu)                                       # decode the mean, no sampling
fig, ax = plt.subplots(2, 10, figsize=(12, 3.1))
for c in range(10):
    show(ax[0, c], Xte[first_idx[c]])
    show(ax[1, c], rec[c])
    ax[0, c].set_title(f"{c}  (#{first_idx[c]})", fontsize=9)
ax[0, 0].set_ylabel("original")
ax[1, 0].set_ylabel("decoded mu")
fig.supxlabel("digit (test index of the first test image of that digit)", fontsize=10)
fig.supylabel("row", fontsize=10)
fig.suptitle("Figure 2 - first test image of each digit (top) and its reconstruction from mu (bottom), d = 16")
fig.tight_layout()
fig.savefig(OUT / "fig2-reconstructions.png", dpi=150)
plt.close(fig)
# --8<-- [end:fig2]
res["ex1"]["fig2_test_indices"] = first_idx

# --8<-- [start:fig3]
# ---------------------------------------------------------------- Figure 3
with torch.no_grad():
    torch.manual_seed(0)
    z = torch.randn(64, 16)                  # z ~ N(0, I)
    samples = m16.decode(z)                  # sigmoid outputs (pixel means), not binarised
fig, ax = plt.subplots(figsize=(6.4, 6.8))
ax.imshow(tile(samples, 8, 8), cmap="gray", vmin=0, vmax=1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Figure 3 - 64 decoded prior samples z ~ N(0, I), d = 16")
ax.set_xlabel("sample column (each cell = sigmoid output of the decoder, torch.manual_seed(0))")
ax.set_ylabel("sample row")
fig.tight_layout()
fig.savefig(OUT / "fig3-prior-samples.png", dpi=150)
plt.close(fig)
# --8<-- [end:fig3]

# --8<-- [start:diag1]
# ---------------------------------------------------------------- diagnostics for 1E
with torch.no_grad():
    mu_te, lv_te = m16.encode(Xte)
    mu_va, _ = m16.encode(Xva)
    kl_dim = (-0.5 * (1 + lv_te - mu_te ** 2 - lv_te.exp())).mean(0)          # KL per latent dim
    zp = torch.randn(10000, 16, generator=torch.Generator().manual_seed(7))   # fresh prior draws
    # where do prior samples land? distance to the nearest encoded test mean,
    # compared with the same distance for encoded validation images
    nn_prior = torch.cdist(zp, mu_te).min(1).values
    nn_val = torch.cdist(mu_va, mu_te).min(1).values
    # what does the decoder output? share of "grey" pixels (0.1 < value < 0.9)
    grey = lambda p: ((p > 0.1) & (p < 0.9)).float().mean().item()
    # decode mu vs decode one sampled z (the Figure 2 choice)
    bce = lambda logits, x: (F.softplus(logits) - x * logits).sum(1)
    zs = m16.reparameterize(mu_te, lv_te, torch.Generator().manual_seed(1))
    rec_mu = bce(m16.decode_logits(mu_te), Xte)
    rec_z = bce(m16.decode_logits(zs), Xte)
res["ex1"]["diag"] = dict(
    kl_per_dim_test=kl_dim.tolist(),
    active_dims_kl_gt_0_1=int((kl_dim > 0.1).sum()),
    mean_posterior_sigma=torch.exp(0.5 * lv_te).mean().item(),
    nn_dist_median_prior_to_test_mu=nn_prior.median().item(),
    nn_dist_median_val_mu_to_test_mu=nn_val.median().item(),
    frac_prior_farther_than_val_p95=(nn_prior > nn_val.quantile(0.95)).float().mean().item(),
    grey_frac_test_images=grey(Xte),
    grey_frac_recon_mu=grey(m16.decode(mu_te)),
    grey_frac_prior_samples=grey(m16.decode(zp)),
    recon_test_decode_mu=rec_mu.mean().item(),
    recon_test_decode_sampled_z=rec_z.mean().item(),
    frac_images_mu_better=(rec_mu < rec_z).float().mean().item(),
)
# --8<-- [end:diag1]
print("diag ex1", {k: v for k, v in res["ex1"]["diag"].items() if k != "kl_per_dim_test"}, flush=True)

# --8<-- [start:fig4]
# ---------------------------------------------------------------- Figure 4
with torch.no_grad():
    mu2_te = m2.encode(Xte)[0].numpy()
    mu2_va = m2.encode(Xva)[0].numpy()
lab = yte.numpy()
fig, ax = plt.subplots(figsize=(8, 6.6))
cmap = plt.get_cmap("tab10")
for c in range(10):
    s = lab == c
    ax.scatter(mu2_te[s, 0], mu2_te[s, 1], s=4, color=cmap(c), alpha=.6, label=str(c), rasterized=True)
ax.set_title("Figure 4 - posterior means mu of all 10,000 test images, d = 2")
ax.set_xlabel("mu_1 (latent dimension 1)")
ax.set_ylabel("mu_2 (latent dimension 2)")
ax.grid(alpha=.3)
ax.legend(title="digit", markerscale=4, loc="center left", bbox_to_anchor=(1.01, .5))
fig.tight_layout()
fig.savefig(OUT / "fig4-latent-scatter.png", dpi=150)
plt.close(fig)
# --8<-- [end:fig4]

# --8<-- [start:fig5]
# ---------------------------------------------------------------- Figure 5
q = np.linspace(0.05, 0.95, 15)
zv = norm.ppf(q)                                   # z = Phi^-1(q)
zz = torch.tensor([[z1, z2] for z2 in zv[::-1] for z1 in zv],    # row 0 = largest z_2
                  dtype=torch.float32)
with torch.no_grad():
    grid = m2.decode(zz)
fig, ax = plt.subplots(figsize=(8, 8.3))
ax.imshow(tile(grid, 15, 15), cmap="gray", vmin=0, vmax=1, extent=[0, 15, 0, 15])
ticks = np.arange(15) + 0.5                        # cell centres; y = 0 is the bottom row
labels = [f"{round(v, 2) + 0.0:.2f}" for v in zv]       # + 0.0 turns -0.00 into 0.00
ax.set_xticks(ticks, labels, rotation=90, fontsize=8)
ax.set_yticks(ticks, labels, fontsize=8)          # bottom row = smallest z_2
ax.set_xlabel("z_1 = Phi^-1(q),  q = 0.05 ... 0.95")
ax.set_ylabel("z_2 = Phi^-1(q),  q = 0.05 ... 0.95")
ax.set_title("Figure 5 - decoded 15 x 15 grid of the 2-D latent plane, d = 2")
fig.tight_layout()
fig.savefig(OUT / "fig5-latent-grid.png", dpi=150)
plt.close(fig)
# --8<-- [end:fig5]
res["ex2"]["grid_z_values"] = zv.tolist()

# --8<-- [start:diag2]
# ---------------------------------------------------------------- diagnostics for 2E
# overlap in Figure 4: 15-NN classifier fit on validation mu, applied to test mu
knn = KNeighborsClassifier(n_neighbors=15).fit(mu2_va, yva.numpy())
pred = knn.predict(mu2_te)
conf = np.zeros((10, 10), int)
np.add.at(conf, (lab, pred), 1)
pairs = sorted(((conf[a, b] + conf[b, a], a, b) for a in range(10) for b in range(a + 1, 10)), reverse=True)
# which digit each Figure-5 cell falls next to (same 15-NN rule), row 0 = top
grid_lab = knn.predict(zz.numpy()).reshape(15, 15)
# per-digit cost of d = 2 versus d = 16 (decode mu, test set)
with torch.no_grad():
    r16 = bce(m16.decode_logits(m16.encode(Xte)[0]), Xte)
    r2 = bce(m2.decode_logits(m2.encode(Xte)[0]), Xte)
res["ex2"]["diag"] = dict(
    knn15_test_accuracy=float((pred == lab).mean()),
    top_confused_pairs=[dict(digits=[int(a), int(b)], count=int(n)) for n, a, b in pairs[:5]],
    grid_nearest_digit_rows_top_to_bottom=["".join(map(str, r)) for r in grid_lab],
    recon_mu_per_digit_d16=[r16[yte == c].mean().item() for c in range(10)],
    recon_mu_per_digit_d2=[r2[yte == c].mean().item() for c in range(10)],
    grey_frac_grid=grey(grid),
)
# --8<-- [end:diag2]
print("diag ex2", res["ex2"]["diag"], flush=True)

# --8<-- [start:fig6]
# ---------------------------------------------------------------- Figure 6
# Pair rule (fixed before looking at Figure 6): the first test image of digit 0 and the
# first test image of digit 1 -- the two leftmost columns of Figure 2, so both endpoints'
# reconstructions are already on display. A 0 and a 1 share almost no ink, so a pixel
# blend of them cannot look like either digit.
ia, ib = first_idx[0], first_idx[1]
t = torch.linspace(0, 1, 10)
xa, xb = Xte[ia], Xte[ib]
pixel_row = torch.stack([(1 - s) * xa + s * xb for s in t])
with torch.no_grad():
    mua, mub = m16.encode(Xte[[ia, ib]])[0]                     # d = 16 model
    latent_row = m16.decode(torch.stack([(1 - s) * mua + s * mub for s in t]))
fig, ax = plt.subplots(2, 10, figsize=(12, 3.3))
for j in range(10):
    show(ax[0, j], pixel_row[j])
    show(ax[1, j], latent_row[j])
    ax[0, j].set_title(f"t = {t[j]:.2f}", fontsize=9)
ax[0, 0].set_ylabel("pixel space", fontsize=9)
ax[1, 0].set_ylabel("latent (d=16)", fontsize=9)
fig.supxlabel("interpolation weight t (t = 0: test #%d, t = 1: test #%d)" % (ia, ib), fontsize=10)
fig.supylabel("space", fontsize=10)
fig.suptitle(f"Figure 6 - interpolation from test #{ia} (digit 0) to test #{ib} (digit 1)")
fig.tight_layout()
fig.savefig(OUT / "fig6-interpolation.png", dpi=150)
plt.close(fig)
# --8<-- [end:fig6]

# --8<-- [start:diag3]
# how digit-like is each frame? A proxy only: a multinomial logistic regression
# trained on the 50,000 training images, applied to every frame of Figure 6
clf = LogisticRegression(max_iter=300).fit(Xtr.numpy(), ytr.numpy())
frames = {"pixel": pixel_row.numpy(), "latent": latent_row.numpy()}
res["ex2"]["interp"] = dict(
    index_a=ia, digit_a=int(yte[ia]), index_b=ib, digit_b=int(yte[ib]), t=t.tolist(),
    endpoint_recon_nats=[r16[ia].item(), r16[ib].item()],
    endpoint_recon_percentile_in_digit=[
        (r16[yte == yte[i]] <= r16[i]).float().mean().item() for i in (ia, ib)],
    logreg_val_accuracy=float(clf.score(Xva.numpy(), yva.numpy())),
    **{f"{k}_pred": clf.predict(v).tolist() for k, v in frames.items()},
    **{f"{k}_maxprob": clf.predict_proba(v).max(1).round(3).tolist() for k, v in frames.items()},
    **{f"{k}_grey_frac": [grey(torch.tensor(f)) for f in v] for k, v in frames.items()},
)
for k, v in frames.items():                       # summaries over the 8 inner frames (0 < t < 1)
    res["ex2"]["interp"][f"{k}_inner_mean_maxprob"] = float(clf.predict_proba(v[1:-1]).max(1).mean())
    res["ex2"]["interp"][f"{k}_inner_mean_grey_frac"] = grey(torch.tensor(v[1:-1]))
# --8<-- [end:diag3]
print("interp", res["ex2"]["interp"], flush=True)

with open(OUT / "results.json", "w") as f:
    json.dump(res, f, indent=1)
print("saved", OUT / "results.json", f"in {time.time() - T0:.0f} s")
