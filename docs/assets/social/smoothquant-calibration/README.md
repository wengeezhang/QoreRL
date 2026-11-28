# SmoothQuant calibration comparison

Published on X on 2026-11-28: [comparison post](https://x.com/wengeezhang/status/2126673115979948106).

Two matching diagrams explain the calibration modes documented in the [calibration walkthrough](../../../smoothquant-calibration.md):

- **Local calibration:** [PNG](local-projection.png) · [editable SVG](local-projection.svg). Replay the current recipe, capture projection inputs before their local transform, and compare each projection independently against a local BF16 reference.
- **Full-model logits calibration:** [PNG](model-logits.png) · [editable SVG](model-logits.svg). Fit from full-model BF16 activations and evaluate the entire candidate set against BF16 logits at the same held-out token positions.

The diagrams show design and control flow, not benchmark results. Implementation and CPU tests were complete at publication; native GPU validation remained pending. S means SmoothQuant smoothing factors. Layer labels refer to their `down_proj` projections. The BF16 reference executes the current weights; it does not reconstruct an all-BF16 training history.

The SVG files are the editable sources. PNG exports are 2100 × 2460 pixels, rendered with `@resvg/resvg-js`; the existing QoreRL mark is embedded unchanged.

## Published text

> I added full-model logits validation to QoreRL's SmoothQuant calibration.
>
> Local checks can miss interactions between layers. The new mode fits from BF16 activations, then tests the whole candidate together.
>
> CPU-tested; GPU validation next.
>
> https://github.com/wengeezhang/QoreRL/blob/main/docs/smoothquant-calibration.md
