# FIGURES TO DRAW — placeholders only, art not made yet (2026-06-20)

The thesis is light on diagrams vs. e.g. IAD-R1 (their Fig. 1 fuses dataset composition +
two-stage training pipeline + a qualitative example + result panels into one overview). We want
more system/methodology diagrams. **For now we only insert compile-safe PLACEHOLDER boxes** (red-
bordered `figure` blocks with a caption + `\label`), one per planned figure, placed as we walk
through each chapter together. Grep the thesis for `FIGURE PLACEHOLDER` to find them all.

## Planned figures (decide final placement during the chapter walkthrough)
| # | Where | Figure | Notes / analog |
|---|---|---|---|
| F-overview | Ch.1 Introduction | **System overview** — Real-IAD → Gemini trace generation (AnomalyThink) → SFT → GRPO → eval on MMAD (DS-MVTec/VisA). The flagship one-page diagram. | direct analog of IAD-R1 Fig. 1 |
| F-tracegen | Ch.3 Dataset & Traces | **Trace-generation pipeline** — 6-phase prompt, dual-image input (image + RED GT-mask overlay), Gemini-2.5-Flash, auto-reject + schema QA, ShareGPT corpus. | |
| F-traceexample | Ch.3 | **Annotated AnomalyThink trace** — the structured `<think>/<type>/<location>/<answer>` output on one example (NG + OK). | analog of IAD-R1 bottom-left example |
| F-sftarch | Ch.4 SFT | **Frozen-ViT architecture + four-factor grid** — what is trained (LLM+projector) vs frozen (ViT); the 3B/7B × frozen/unfrozen × 6K/15K design. | |
| F-grpo | Ch.5 GRPO | **GRPO + reward schematic** — group sampling (G=4), the accuracy+consistency reward decomposition, KL-to-frozen-init (β). | |
| F-graphabstract | Abstract (optional) | small graphical abstract / teaser (often omitted in TUD style; overview in Ch.1 may suffice). | |

Results chapter already has plots (per-product bars, reward/KL curves, BA bars) — likely enough there.

## Placeholder LaTeX pattern (self-contained, no preamble change, compiles)
```latex
\begin{figure}[htbp]\centering
\fcolorbox{accentred}{accentred!4}{\parbox{0.9\linewidth}{\centering\rule{0pt}{3.2cm}%
  \textbf{\color{accentred}[FIGURE PLACEHOLDER --- to be drawn]}\\[4pt]\itshape <one-line description>\par\vspace{8pt}}}
\caption[<short>]{\textbf{[Placeholder]} <description>.}
\label{fig:<name>}
\end{figure}
```
