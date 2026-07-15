# NOTES

Final config: 4-layer/4-head, n_embd 160, block 128, 1,892,800 params, BPE tokenizer
(vocab 2048, trained on the corpus, byte fallback for losslessness), untied weights, RoPE
instead of learned position embeddings, Muon optimizer on the hidden 2D weights (AdamW on
everything else), WSD (warmup-stable-decay) learning-rate schedule, batch 32, 2000 steps.
Dev bpb: 2.3718 (baseline) → **1.7328** (final, −27% relative).

The single biggest win was swapping the byte-level tokenizer for a corpus-trained BPE:
Devanagari costs 3 bytes/char under byte tokenization, so BPE roughly triples the effective
context per 2000-step budget on the Hindi half of the corpus. The second-biggest win was
raising batch size (more distinct tokens seen per step matters more than gradient-noise
reduction when steps, not compute, are the binding constraint). On top of that base, we
tested RoPE, Muon, WSD, and QK-Norm as isolated one-variable-at-a-time changes; RoPE, Muon,
and WSD each won on their own, QK-Norm was a wash. The final run stacks the three winners
together (RoPE + Muon + WSD) rather than shipping just the best isolated run, and stacking
them worked cleanly: no interference, and the combined run beat every individual one. See
RUNLOG.md for the full run-by-run trail, including the losers and why they lost.
