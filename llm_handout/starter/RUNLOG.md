# RUNLOG

One entry per run: what we changed, why we thought it'd help, what happened.

---

## Run 0 - baseline
Change: none, ran the starter as-is (batch 8, constant lr 3e-4, byte tokenizer, flat init).
Result: dev bpb **2.3718**. Loss was still falling hard at step 2000 - the run is schedule-starved, not out of capacity. Also noticed the byte tokenizer burns 3 bytes per Devanagari character, so Hindi text eats way more of the context window than it should.

Biggest levers we picked out from this: tokenizer (BPE), LR schedule (warmup + decay), batch size.

---

## EDA notes (before Run 1)
Corpus is 85.8% ASCII / 14.1% Devanagari / 0.1% other by char count, but Devanagari is 3 bytes/char so it's ~33% of bytes. A BPE vocab of 2048 trained on the corpus drops Devanagari to ~1 token/char - roughly 3x more real text per 128-token window. Loss curve confirms Run 0 hadn't converged. Init variance grows 15x across the 4 blocks under the baseline's flat init, which is exactly what GPT-2-style scaled init is supposed to fix (tried in Run 2, didn't pan out - see below). Gradient-norm variance barely changes between batch 8 and 64, so batch size looked like a low-confidence lever going in (it wasn't, see Run 6).

Sources checked: modded-nanogpt (Keller Jordan), a NanoGPT speedrun field guide and worklog, an Indic multilingual tokenizer paper (arXiv:2511.03237), a sub-byte BPE paper (arXiv:2506.07541), the muP paper (arXiv:2203.03466), Chinchilla/Gopher for the warmup+cosine recipe.

---

## Run 1 - LR warmup + cosine decay + AdamW + grad clip
Change: peak lr 3e-4 → 1.5e-3 with 100-step warmup and cosine decay to 10%, switched Adam → AdamW (wd 0.1), added grad clip 1.0.
Hypothesis: Run 0 was leaving progress on the table from schedule alone, not architecture.
Result: dev bpb 2.3718 → **2.1904** (−7.6%). Biggest single win so far. New base for everything after.

---

## Run 2 - GPT-2/muP-style scaled init - rejected
Change: init std 0.05 → 0.02, with residual-projection layers additionally scaled by 1/sqrt(2·n_layer).
Hypothesis: fixes the 15x activation-variance growth seen in EDA.
Result: worse at every LR we tried (swept 6e-4/1.5e-3/3e-3, all lost to flat init). Best of sweep still 2.1904 → 2.3338.
Why it lost: this init is built for much deeper nets trained much longer. At 4 layers and 2000 steps, the larger flat-init gradients seem to just get AdamW further, faster. Reverted.

---

## Run 3 - weight tying (byte vocab 256) - rejected
Note: mid-run, a different concurrent process directly hand-edited model.py/tokenizer.py outside this log - reverted before continuing so results stay comparable.
Change: tie_weights True (frees 40,960 params at vocab 256).
Result: dev bpb 2.1904 → 2.2203, worse.
Why: the freed params are too small a slice of the budget to matter, and sharing the embedding matrix seems to constrain optimization more than it helps in 2000 steps. Left untied.

---

## Run 4 - byte-level BPE tokenizer, vocab 2048
Change: replaced the byte tokenizer with a from-scratch BPE (pure stdlib, byte-fallback base vocab so it's lossless by construction). Pre-tokenizer splits by Unicode category rather than regex `\w`, because `\w` doesn't match Devanagari matras/virama and was badly fragmenting Hindi text in an early prototype.
Hypothesis: collapsing Devanagari toward ~1 token/char should let the model see much more real text per 128-token window.
Result: dev bpb 2.1904 → **1.9967** (−8.8%), the biggest win in the project. Both ASCII and Devanagari per-script bpb improved (we'd worried it might trade one off against the other - it didn't). n_params up to 1,913,280 (bigger vocab → bigger embed/head tables), still comfortably under cap.

---

## Run 5 - weight tying re-test (BPE vocab 2048) - rejected again
Change: tie_weights True, now on top of the BPE tokenizer where tying frees a much bigger chunk (327,680 params, ~17% of total).
Hypothesis: bigger vocab means tying should matter more this time.
Result: dev bpb 1.9967 → 2.0257, worse again, same direction as Run 3.
Why: same story as before - freeing params without reinvesting them doesn't help; forcing a shared embedding/head matrix costs more than the savings are worth in this short a run. Left untied for good.

---

## Run 6 - batch size 8 → 32
Change: batch 8 → 32, everything else same as Run 4.
Hypothesis: we're step-capped, not compute-capped, so more tokens per step should help even though the EDA's gradient-noise proxy was unconvincing.
Result: dev bpb 1.9967 → **1.7863** (−10.5%). Biggest single-lever win after the tokenizer. New best config.

---

## Literature review round 2 (after Run 6)
Went back through the external review notes and checked every citation against the actual paper before queuing anything, since these notes have gotten paper mechanisms wrong before.

- **SOAP**: real optimizer, but the review note's description was garbled (it's Shampoo-in-Adafactor's-eigenbasis, not "Schur-power" anything). Skipped - much higher implementation risk than Muon for the same basic hypothesis, and Muon was already queued.
- **NorMuon** (arXiv:2510.05491): checks out - row-wise second-moment normalization after Muon's orthogonalization step, fixes uneven per-neuron update norms. Queued as Run 12, conditional on Muon actually winning first.
- **QK-norm paper** (arXiv:2511.21377): re-read carefully - its real contribution is an alternative to QK-norm for Multi-Latent Attention specifically, and it says plain RMSNorm QK-norm (what we're already testing in Run 7) is the right choice for ordinary multi-head attention. Confirms Run 7's approach, nothing to change.
- **QK-Normed MLA** (arXiv:2606.16310): real, but solves a caching problem specific to MLA, which we don't use. Not applicable.
- **Muennighoff et al.** (arXiv:2305.16264): real and relevant - repeating data up to 4 epochs costs almost nothing, meaningful gains keep coming until ~16 epochs. We're nowhere near either boundary at our step/batch sizes.
- **Zuo et al.** (arXiv:2607.04969, memorization-guided data reuse): real, but a lot more engineering than anything else in this project for uncertain payoff at our scale. Parked as a stretch idea, not run.
- **NanoGPT Slowrun**: real community benchmark, but its gains come from large-model ensembling well outside our param cap. Not directly portable.

Net: queued Run 12 (NorMuon, conditional on Run 10) and Run 13 (weight decay bump, conditional on seeing train/dev divergence) - nothing else made the cut.

**Correction, later:** re-checked Muennighoff against the actual PDF, not just the abstract. Earlier phrasing above overstated it as a "4-epoch hard ceiling" - the real numbers are: negligible loss difference up to 4 epochs, but gains keep coming until ~16. The paper also doesn't test regularization as a fix for repeated-data degradation - that's an untested aside from the authors, not a result. So Run 11 at ~3.5 epochs was never actually near a danger zone, and Run 13 stayed correctly gated on an observed symptom rather than becoming "required" prep.

---

## Run 9 - WSD schedule
Change: cosine decay → warmup-stable-decay (constant lr, then a sharp decay only in the last 10%), on top of Run 6.
Hypothesis: keeping lr high longer before a short sharp decay should beat gradual cosine decay.
Result: dev bpb 1.7863 → **1.7575** (−1.6%). New best.

---

## Run 7 - QK-Norm
Change: RMSNorm on Q/K before the attention dot product, on top of Run 6 (cosine schedule, not WSD - tested independently).
Hypothesis: controls attention-logit growth, should stabilize training.
Result: dev bpb 1.7863 → 1.785 - basically flat (−0.07%, noise-level). Not a regression, but not worth adopting on its own. Worth retrying at a higher LR sometime, since QK-norm's whole point is that it should let you push LR further, and we never actually tested that.

---

## Run 8 - RoPE
Change: rotary position embeddings instead of the learned absolute position table, on top of Run 6.
Hypothesis: standard nanoGPT-speedrun swap, should be free params (no more `block_size * n_embd` table) and better generalization within the context window.
Result: dev bpb 1.7863 → **1.7554** (−1.7%). New best at the time - beat both Run 9 and Run 6. Also 20,480 fewer params.

---

## Run 10 - Muon optimizer
Change: Muon (Newton-Schulz orthogonalized updates) on the >=2D hidden weights, AdamW everywhere else, on top of Run 6.
Hypothesis: orthogonalized updates avoid the low-rank bias plain Adam has on matrix params - should matter a lot under a tight step budget.
Result: dev bpb 1.7863 → **1.7374** (−2.7%). Biggest single win of the whole isolated-technique round, beats RoPE and WSD too.

At this point the deadline got tighter and we stopped testing things in isolation. Run 11 (batch 64) was killed mid-run, unscored, to free the CPU - not a loss, just not worth finishing given the time left.

---

## Final run - RoPE + Muon + WSD stacked
Each of these three was already tested on its own, individually, before this run: Run 8 (RoPE only, bpb 1.7554), Run 10 (Muon only, bpb 1.7374), Run 9 (WSD only, bpb 1.7575) - all three beat the Run 6 base (1.7863) independently. This run is the first time they're combined.
Change: `--rope 1 --optimizer muon --schedule wsd`, batch 32, on top of Run 6.
Hypothesis: they're independent changes (positional encoding, optimizer, LR schedule) touching different parts of the system, so should stack rather than fight each other.
Result: dev bpb **1.7328** - beats every individual run (Muon alone 1.7374, RoPE alone 1.7554, WSD alone 1.7575, base 1.7863). n_params 1,892,800, steps 2000, both caps satisfied. The hypothesis held: none of the three fought each other, and stacking beat every one of them on its own. This is the final submitted checkpoint (`ckpt.pt`).
