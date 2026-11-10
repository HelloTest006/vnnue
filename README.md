# VNNUE: Very Easily Updatable Neural Networks

**An exact, formally verified update calculus and a power-sum interaction channel for chess evaluation.**

📄 Paper: [`paper/vnnue.pdf`](paper/vnnue.pdf) (36 pages)

NNUE made neural evaluation affordable inside alpha-beta chess engines by updating the first layer incrementally on every move. This project treats that mechanism as an algebraic object. For a fixed weight table, the first layer is a group homomorphism ℤᴺ → (ℤ/2¹⁶)ᴸ. Incremental updates, Finny-table refreshes, Stockfish's hybrid king update, lazy catch-up and wrap-around integer arithmetic are all correct because of identities of that homomorphism. The project makes those identities explicit, machine-checks them in Lean 4, and uses them to design **VNNUE**, a proposed extension of Stockfish's K32H3 network. VNNUE adds:

- **PSI (power-sum interaction) channel.** A small, refresh-free accumulator, shared by both perspectives, of first and second power sums of a learned piece embedding. By the Newton–Girard identities it yields exact all-pairs piece×piece interaction terms (a factorization machine).
- **Exact source selection.** Accumulator rebuilds choose from a candidate set that contains Stockfish's own. The choice is never worse at a single event, given the same state and the stated cost model.
- **Optional modes.** Low-rank king transitions, net-delta catch-up, and symmetry-equivariant views.

Setting the new weights to zero recovers the NNUE evaluation exactly.

## Honest status

| | Status |
|---|---|
| Correctness of all update rules | **Proved** in Lean 4 + Mathlib: 149 theorems, no `sorry`, axioms only `propext` / `Classical.choice` / `Quot.sound` |
| Bit-exactness in practice | **Measured**: 14.5 M checks on real Lichess games and search trees, 0 mismatches |
| PSI overhead | **Measured**: ≈1–5 % extra memory traffic, within the proved bound |
| Source-selection savings | **Small**: ≈6 % fewer column ops vs. a modelled Stockfish rule on games, almost all of it from taking the argmin over Stockfish's *own* candidates |
| Net-delta catch-up (greedy) | **Negative**: +20 % columns on toy search traces |
| Low-rank king transitions | **Rarely useful** at default settings |
| PSI accuracy | Learns pair structure on synthetic targets; **no benefit observed** on small real-label data |
| Playing strength (Elo), engine nps | **Not tested.** This needs a C++/SIMD Stockfish implementation, large-scale training with QAT, and SPRT games |

## Repository layout

| Path | Contents |
|---|---|
| [`paper/`](paper/) | LaTeX sources, `refs.bib`, `vnnue.bbl`, figures, compiled `vnnue.pdf`; `modal_tex.py` (compile on Modal), `make_arxiv.py` (build and verify the arXiv bundle) |
| [`lean/`](lean/) | Lean 4 development: `VNNUE/{Basic,Accumulator,DeltaMonoid,Quantization,Interaction,Symmetry,Refresh,Complexity}.lean`; `modal_check.py` (build + axiom audit on Modal) |
| [`full_check.log`](full_check.log) | Log of the full Lean build and the axiom audit of all 149 theorems |
| [`experiments/`](experiments/) | Python prototype: scripts, derived datasets (`data/*.npz`), raw results (`results/*.json`), figures, logs, and [`RESULTS.md`](experiments/RESULTS.md) |
| [`design/`](design/) | Working design documents: three candidate designs and the merged, judge-corrected `SPEC.md`. These are historical and partly superseded by the Lean sources and the paper |
| [`research/`](research/) | Literature notes used to write the paper. Items marked `[UNVERIFIED]` were not checked |

## Verifying the proofs

Lean `v4.35.0`, Mathlib `v4.35.0` (pinned in `lean/lakefile.toml` and `lean/lake-manifest.json`).

**Locally** (needs [elan](https://github.com/leanprover/elan) and about 8 GB of free RAM):

```bash
cd lean
lake exe cache get   # download prebuilt Mathlib
lake build           # builds all 8 modules
```

**On a Modal CPU instance** (no local Mathlib needed; requires a [Modal](https://modal.com) account):

```bash
cd lean
modal run modal_check.py                                     # full build + axiom audit, prints RESULT: OK
modal run modal_check.py --files VNNUE/Basic.lean,VNNUE/Interaction.lean
```

The checker prints `#print axioms` for every theorem and fails if any theorem depends on `sorryAx`.

## Reproducing the experiments

Python (tested with 3.14) with `numpy`, `torch`, `matplotlib` and `python-chess`. The run times below are from a 4-core laptop.

```bash
cd experiments
# Optional: regenerate the derived datasets. This needs the raw CC0 Lichess file
# data/lichess/lichess_db_standard_rated_2013-01.pgn.zst (from https://database.lichess.org).
python gen_data.py          # ~8 min
python e1_exactness.py      # ~35 min
python e2_e4_opcounts.py    # ~53 min
python e2_threat.py         # ~1 min
python e3_cache_reset.py    # ~11 min
python train.py             # ~60 min
python bench.py             # ~3 min
python make_report.py       # regenerates tables, figures and RESULTS.md
```

The derived `data/*.npz` files are committed, so every script except `gen_data.py` runs without the raw PGN file. Seeds are fixed.

## Building the paper

```bash
cd paper
modal run modal_tex.py     # latexmk + BibTeX on Modal -> vnnue.pdf, vnnue.bbl
modal run make_arxiv.py    # -> ../arxiv/vnnue-arxiv.tar.gz, verified with pdflatex only
```

With a local TeX Live: `latexmk -pdf vnnue.tex`.

## Use of AI tools

The research notes, design, Lean formalization, experiment code and paper drafts were produced with substantial assistance from an AI system (Claude, by Anthropic), working under the author's direction. Every theorem the paper cites as proved is checked by the Lean kernel, and every reported number comes from the scripts in this repository.

## License

- **Code** (Lean sources, Python scripts, build scripts): [MIT](LICENSE)
- **Text and figures** (paper, `RESULTS.md`, research and design notes): [CC BY 4.0](LICENSE-CC-BY-4.0.txt)
- The derived datasets come from the Lichess database, which is released under CC0.

## Citation

See [`CITATION.cff`](CITATION.cff), or:

```bibtex
@misc{vnnue2026,
  author       = {HelloTest006},
  title        = {{VNNUE}: Very Easily Updatable Neural Networks --- An Exact, Formally Verified Update Calculus and a Power-Sum Interaction Channel for Chess Evaluation},
  year         = {2026},
  howpublished = {\url{https://github.com/HelloTest006/vnnue}}
}
```
