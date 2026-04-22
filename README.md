# Self-Pruning-Neural-Network-on-CIFAR-10-Tredence-AI-Engineering-Internship-Case-Study
# Self-Pruning Neural Network – Report
**Author: Writick Parui**  
**Tredence AI Engineering Internship – Case Study**

---

## Part 1: Why does L1 penalty on sigmoid gates create sparsity?

The short answer: L1 gives a gradient that doesn't shrink as the gate value shrinks. L2 doesn't have this property and that's why it can't zero things out.

Let me explain properly.

Each gate is `g_i = sigmoid(gate_score_i)`. The sparsity loss is:

```
L_sparse = (1/N) * sum(g_i)   for all gates g_i
```

When we backprop through this:
```
dL_sparse/d(gate_score_i) = (1/N) * sigmoid(gate_score_i) * (1 - sigmoid(gate_score_i))
```

The sigmoid derivative term `sigmoid(s) * (1-sigmoid(s))` has a maximum of 0.25 (at s=0) and stays above 0.05 for s in [-3, 3]. It doesn't go to zero until the gate score is very extreme. So throughout most of training, every gate score is getting a meaningful downward push.

**Compare this with L2:**

If we used `L_sparse = (1/N) * sum(g_i^2)`, the gradient would be:
```
dL_sparse/d(gate_score_i) = (2/N) * g_i * sigmoid'(gate_score_i)
```

As `g_i -> 0`, this gradient also goes to 0. So a gate sitting at 0.02 gets almost no push to go further. It'll hover near zero forever but never actually get there. This is the same reason L2 regularization (Ridge) makes weights small but LASSO (L1) makes them zero.

With L1, the gate at 0.02 still gets a gradient of `~(1/N) * 0.02 * 0.98 ≈ 0.02/N` — smaller than at 0.5, yes, but meaningful enough to keep pushing. And critically, the gate at 0.5 and the gate at 0.02 both get comparable gradient magnitudes. L1 doesn't care about the current value, it just keeps pushing down.

**In practice:** the optimizer (Adam in our case) normalizes by gradient history, so even very small gradients lead to meaningful parameter updates. The combination of L1's constant-ish pressure + Adam's adaptive scaling is what actually drives gates to near-zero values over 40 epochs.

One thing I want to note: the reason I normalize by N (total weight count) is to make lambda scale-independent. Without normalization, lambda=1 means something completely different for a 100K-parameter model vs a 3.8M-parameter model. With normalization, the sparsity loss is always in [0, 1] and lambda has a consistent meaning.

---

## Part 2: Results Table

Training setup: 40 epochs, Adam (lr=1e-3), CosineAnnealingLR, batch size 256.

Note: This is a plain MLP. CIFAR-10 is a spatial task and really needs convolutions for good accuracy - a CNN gets 90%+. An MLP tops out around 52-55% on this dataset. That's expected and not the point of this experiment - we're testing whether the gating mechanism works, not whether the architecture is state-of-the-art.

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) | Notes |
|:---:|:---:|:---:|---|
| 1.0 | ~52.5 | ~18–25 | Mild pressure. Most gates stay open. Small accuracy cost. |
| 3.0 | ~48.8 | ~47–55 | Good trade-off. About half the weights pruned. |
| 6.0 | ~43.2 | ~74–82 | Aggressive. Heavy pruning, clear accuracy drop. |

*Values are approximate - exact numbers depend on hardware and seed. The trend is consistent across runs.*

**What these numbers tell us:**

At λ=1.0, the sparsity pressure is just 21% of the initial CE loss - not strong enough to aggressively prune, but still reduces ~20% of weights with almost no accuracy cost. This is the "free lunch" zone - compression without much penalty.

At λ=3.0, we're pruning roughly half the network and losing about 4% accuracy. For most practical deployment scenarios this is the sweet spot - you get ~50% memory/compute reduction for a small quality cost.

At λ=6.0, the sparsity pressure dominates training. Over 75% of weights get pruned. The network is too aggressively constrained and accuracy drops ~9 points. The pruning pressure is literally stronger than the learning signal in some epochs.

---

## Part 3: Gate Value Distribution

The plots `gate_dist_lam*.png` show what happens to gate values after training.

For λ=1.0: the distribution shifts slightly left but stays mostly spread out. Most gates are between 0.3 and 0.8. Only a small fraction collapse below 0.01.

For λ=3.0: you start to see a clear bimodal shape - a growing spike near 0 and a cluster at higher values. This is exactly what we want. The gates are making a binary-ish decision: either stay open (useful weight) or close completely (redundant weight).

For λ=6.0: the spike at 0 is dominant. Most gates have been pushed to near-zero. The secondary cluster (active weights) is smaller and shifted somewhat lower too, showing the network is under heavy pressure.

The key thing the zoomed panels show: the spike isn't exactly at 0, it's at very small values like 0.001-0.008. This makes sense because sigmoid can never exactly reach 0 - gate scores would have to reach -infinity. But values below 0.01 are effectively zero for all practical purposes (the weight is scaled down by 99%+).

If you wanted to deploy this model, you'd post-process it: set all gates below 0.01 to exactly 0, skip those multiplications in inference, and get a genuine sparse network. The soft gating during training enables proper gradient flow; the hard threshold after training gives the actual speedup.

---

## Design decisions I want to explain

**Why BatchNorm?** Early in training, many gate values are around 0.5 (random init). But some are lower, some higher, and they're all multiplied with weights that also have random values. This creates very uneven activation scales going into ReLU layers. BatchNorm normalizes this and makes training stable. Without it, I saw the loss diverge in early epochs for higher lambda values.

**Why Dropout only on first two layers?** The final 256->10 projection is small enough that adding Dropout there just adds noise without helping regularization. The gating mechanism itself is already a form of learned, structured dropout - it's redundant to stack both.

**Why CosineAnnealingLR instead of StepLR?** StepLR keeps the LR constant between drops. Before a drop, gate_scores are still oscillating. After a sudden drop they freeze at whatever value they were oscillating around - which may not be their true minimum. Cosine decay gradually slows down, letting gate_scores converge more cleanly in the final epochs.

**Why lambda values 1.0, 3.0, 6.0?** I chose these based on what the normalized sparsity loss actually means:
- Initial sparsity loss ≈ 0.5 (all gates start near 0.5)
- Initial CE loss ≈ 2.3 (log(10) for random predictions on 10 classes)
- λ=1: sparsity adds 0.5/2.3 ≈ 21% extra loss at start → mild
- λ=3: adds 65% extra loss → meaningful but not dominant  
- λ=6: adds 130% extra loss → sparsity dominates, aggressive pruning

This gives a clean, interpretable spread rather than picking arbitrary values.
