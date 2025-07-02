# Critical Layers in Vision Transformers via Logit Lens

This project extends the logit lens technique to Vision Transformers (ViTs) to analyze how internal representations evolve across layers. We evaluate models trained with contrastive learning (CLIP) and self-supervised learning (DINO), using cosine similarity, KL divergence, and true class probability to identify critical layers.

## Shortened Abstract

While the logit lens has been effective for understanding learning dynamics in language models, its application to computer vision remains underexplored. We adapt this technique to ViTs and combine it with three complementary metrics to identify semantically important layers. Our results show that CLIP models develop class-relevant features later than DINO models, highlighting differences in how these training paradigms shape hierarchical representations.

## Usage

Install required packages:

```bash
pip install -r requirements.txt
```

Run notebooks inside the /notebooks directory. Start with:
- logit_lens_description.ipynb for an overview
- logit_lens_CLIP.ipynb and logit_lens_DINO_lp.ipynb for experiments
- results_analysis.ipynb for final comparisons
