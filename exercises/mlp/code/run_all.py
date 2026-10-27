"""Runs Exercises 1-4, saves Figures 1-6 in ../figures and all numbers in results.json."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from mlp import MLP

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(42)          # single rng used throughout the report
R = {}


def standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)      # statistics from the training set only
    return (Xtr - mu) / sd, (Xte - mu) / sd


# ============================================================ Exercise 1 (check)
x = np.array([0.5, -0.2]); y = 1.0
W1 = np.array([[0.3, -0.1], [0.2, 0.4]]); b1 = np.array([0.1, -0.2])
W2 = np.array([0.5, -0.3]); b2 = 0.2; eta = 0.3
z1 = W1 @ x + b1; h1 = np.tanh(z1)
u2 = W2 @ h1 + b2; yh = np.tanh(u2); L = (y - yh) ** 2
dL_dyh = -2 * (y - yh); dL_du2 = dL_dyh * (1 - yh ** 2)
dW2 = dL_du2 * h1; db2 = dL_du2
dh1 = dL_du2 * W2; dz1 = dh1 * (1 - h1 ** 2)
dW1 = np.outer(dz1, x); db1 = dz1
W2n, b2n = W2 - eta * dW2, b2 - eta * db2
W1n, b1n = W1 - eta * dW1, b1 - eta * db1
z1n = W1n @ x + b1n; h1n = np.tanh(z1n); yhn = np.tanh(W2n @ h1n + b2n); Ln = (y - yhn) ** 2
R["ex1"] = {k: np.round(np.asarray(v), 6).tolist() for k, v in dict(
    z1=z1, h1=h1, u2=u2, yhat=yh, L=L, dL_dyhat=dL_dyh, dL_du2=dL_du2, dW2=dW2, db2=db2,
    dh1=dh1, dz1=dz1, dW1=dW1, db1=db1, W2n=W2n, b2n=b2n, W1n=W1n, b1n=b1n,
    L_after=Ln, yhat_after=yhn).items()}

# ================================================================ Exercise 2
centers = np.array([[0., 0.], [3., 3.], [-3., -3.]])
X, c = make_blobs(n_samples=[500, 250, 250], centers=centers, cluster_std=1.2, random_state=42)
y2 = (c > 0).astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y2, test_size=0.2, random_state=42, stratify=y2)

fig, ax = plt.subplots(figsize=(6, 5))
for k, col, lab in [(0, "tab:blue", "Class 0"), (1, "tab:orange", "Class 1")]:
    ax.scatter(X[y2 == k, 0], X[y2 == k, 1], s=10, c=col, alpha=0.7, label=lab)
ax.set(title="Figure 1 - Exercise 2 data (1000 points)", xlabel="$x_1$", ylabel="$x_2$")
ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/fig01-dados-ex2.png", dpi=150); plt.close(fig)

lin = LogisticRegression().fit(Xtr, ytr)
R["ex2_linear_acc"] = float(lin.score(Xte, yte))

Xtr_s, Xte_s = standardize(Xtr, Xte)
cfg2 = dict(sizes=[2, 8, 1], lr=0.5, epochs=2000)
net2 = MLP(cfg2["sizes"], output="sigmoid", rng=rng)
hist2 = net2.fit(Xtr_s, ytr.reshape(-1, 1).astype(float), lr=cfg2["lr"], epochs=cfg2["epochs"])
acc2 = float((net2.predict(Xte_s) == yte).mean())
R["ex2"] = dict(cfg=cfg2, test_acc=acc2, train_acc=float((net2.predict(Xtr_s) == ytr).mean()),
                final_loss=hist2[-1], n_params=net2.n_params())

# gradient check (numerical vs analytical) on a small batch
chk = MLP([2, 5, 1], output="sigmoid", rng=np.random.default_rng(0))
Xc, Yc = Xtr_s[:20], ytr[:20].reshape(-1, 1).astype(float)
dW, db = chk.backward(chk.forward(Xc), Yc)
maxerr = 0.0
for l in range(len(chk.W)):
    for idx in np.ndindex(chk.W[l].shape):
        old = chk.W[l][idx]
        chk.W[l][idx] = old + 1e-5; lp = chk.loss(chk.forward(Xc)[-1], Yc)
        chk.W[l][idx] = old - 1e-5; lm = chk.loss(chk.forward(Xc)[-1], Yc)
        chk.W[l][idx] = old
        maxerr = max(maxerr, abs((lp - lm) / 2e-5 - dW[l][idx]))
R["gradcheck_max_abs_err"] = maxerr

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(hist2, label="Training loss (BCE)")
ax.set(title="Figure 2 - Exercise 2 training loss", xlabel="Epoch", ylabel="Binary cross-entropy")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(f"{FIG}/fig02-loss-ex2.png", dpi=150); plt.close(fig)

# decision boundary over the test points
g = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300)
h = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300)
gx, gy = np.meshgrid(g, h)
grid = np.c_[gx.ravel(), gy.ravel()]
mu, sd = Xtr.mean(0), Xtr.std(0)
zz = net2.predict((grid - mu) / sd).reshape(gx.shape)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.contourf(gx, gy, zz, levels=[-0.5, 0.5, 1.5], colors=["#bcd4f0", "#f8d2a8"], alpha=0.8)
for k, col, lab in [(0, "tab:blue", "Class 0"), (1, "tab:orange", "Class 1")]:
    ax.scatter(Xte[yte == k, 0], Xte[yte == k, 1], s=14, c=col, edgecolor="k", linewidth=0.3, label=f"{lab} (test)")
ax.set(title=f"Figure 3 - MLP decision boundary (test acc. {acc2:.1%})", xlabel="$x_1$", ylabel="$x_2$")
ax.legend(loc="upper left"); fig.tight_layout(); fig.savefig(f"{FIG}/fig03-fronteira-ex2.png", dpi=150); plt.close(fig)

# ================================================================ Exercise 3
X3, y3 = make_classification(n_samples=1500, n_features=4, n_informative=4, n_redundant=0,
                             n_repeated=0, n_classes=3, n_clusters_per_class=2, class_sep=1.0,
                             random_state=42)
X3tr, X3te, y3tr, y3te = train_test_split(X3, y3, test_size=0.2, random_state=42, stratify=y3)
X3tr_s, X3te_s = standardize(X3tr, X3te)
Y3tr = np.eye(3)[y3tr]

R["ex3_linear_acc"] = float(LogisticRegression(max_iter=1000).fit(X3tr, y3tr).score(X3te, y3te))

cfg3 = dict(sizes=[4, 16, 3], lr=0.3, epochs=2000)
net3 = MLP(cfg3["sizes"], output="softmax", rng=rng)
hist3 = net3.fit(X3tr_s, Y3tr, lr=cfg3["lr"], epochs=cfg3["epochs"])
pred3 = net3.predict(X3te_s)
acc3 = float((pred3 == y3te).mean())
cm = confusion_matrix(y3te, pred3)
R["ex3"] = dict(cfg=cfg3, test_acc=acc3, train_acc=float((net3.predict(X3tr_s) == y3tr).mean()),
                final_loss=hist3[-1], n_params=net3.n_params(), confusion=cm.tolist())

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(hist3, label="Training loss (CCE)")
ax.set(title="Figure 4 - Exercise 3 training loss", xlabel="Epoch", ylabel="Categorical cross-entropy")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(f"{FIG}/fig04-loss-ex3.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 5))
ConfusionMatrixDisplay(cm, display_labels=["Class 0", "Class 1", "Class 2"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set(title="Figure 5 - Confusion matrix (test set)", xlabel="Predicted class", ylabel="True class")
fig.tight_layout(); fig.savefig(f"{FIG}/fig05-confusao-ex3.png", dpi=150); plt.close(fig)

# ================================================================ Exercise 4
cfg4 = dict(sizes=[4, 16, 16, 3], lr=cfg3["lr"], epochs=cfg3["epochs"])
seeds = [42, 43, 44, 45, 46]
runs = {"ex3": [], "ex4": []}
curves = {"ex3": [], "ex4": []}
for s in seeds:
    for key, cfg in (("ex3", cfg3), ("ex4", cfg4)):
        n = MLP(cfg["sizes"], output="softmax", rng=np.random.default_rng(s))
        hh = n.fit(X3tr_s, Y3tr, lr=cfg["lr"], epochs=cfg["epochs"])
        runs[key].append(dict(seed=s, test_acc=float((n.predict(X3te_s) == y3te).mean()),
                              train_acc=float((n.predict(X3tr_s) == y3tr).mean()),
                              final_loss=hh[-1], n_params=n.n_params()))
        curves[key].append(hh)
summ = {}
for key in runs:
    summ[key] = dict(runs=runs[key], n_params=runs[key][0]["n_params"],
                     mean_loss=float(np.mean([r["final_loss"] for r in runs[key]])),
                     mean_test=float(np.mean([r["test_acc"] for r in runs[key]])),
                     std_test=float(np.std([r["test_acc"] for r in runs[key]])),
                     mean_train=float(np.mean([r["train_acc"] for r in runs[key]])))
summ["diff_mean_test"] = summ["ex4"]["mean_test"] - summ["ex3"]["mean_test"]
summ["cfg4"] = cfg4; summ["seeds"] = seeds
R["ex4"] = summ

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for key, col, lab in (("ex3", "tab:blue", "Ex. 3 - 1 hidden layer [4,16,3]"),
                      ("ex4", "tab:red", "Ex. 4 - 2 hidden layers [4,16,16,3]")):
    arr = np.array(curves[key])
    for a in arr:
        ax.plot(a, color=col, alpha=0.2, lw=0.8)
    ax.plot(arr.mean(0), color=col, lw=2, label=f"{lab} (mean of {len(seeds)} seeds)")
ax.set(title="Figure 6 - Training loss: 1 vs 2 hidden layers", xlabel="Epoch", ylabel="Categorical cross-entropy")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(f"{FIG}/fig06-comparacao-ex4.png", dpi=150); plt.close(fig)

with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(R, f, indent=2)
print(json.dumps(R, indent=1))
