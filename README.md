# Plivo LLM Speedrun Submission

**Dev bpb: 1.6777** (baseline 2.3718, -29.3%), 1,892,800 params, 2000 optimizer steps.

Deliverables are in [`llm_handout/starter/`](llm_handout/starter/):

- `ckpt.pt` - final checkpoint (2000 steps, 1,892,800 params, dev bpb 1.6777)
- `model.py`, `train.py`, `tokenizer.py`, `muon.py`, `evaluate.py` - code
- `bpe_merges.json` - tokenizer asset, loaded by `tokenizer.py`
- `RUNLOG.md` - every run tried, hypothesis and result
- `NOTES.md` - condensed writeup of the final config
- `SUMMARY.html` - run history and architecture summary

Training/eval data is in `llm_handout/data/` (`train_corpus.txt`, `dev_eval.txt`).

To reproduce or re-score:
```
cd llm_handout/starter
python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt
python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt
```
