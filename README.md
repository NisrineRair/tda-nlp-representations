# TDA-NLP-Representations

*When Annotators Disagree, Topology Explains: Mapper, a Topological Tool for Exploring Text Embedding Geometry and Ambiguity*  (accepted at **EMNLP 2025, Main Conference**).  

We study how fine-tuned encoder-only models represent ambiguous text data.  
Using **Mapper**, a tool from topological data analysis, we show that fine-tuning reshapes embedding space into modular, prediction-aligned regions, even when annotators disagree.  
Our analysis highlights a persistent gap between model confidence and human label ambiguity, demonstrating Mapper’s value as an exploratory tool for representation geometry.

For full details, see the paper: https://aclanthology.org/2025.emnlp-main.426/ 


## Dataset
We use the **MD-Offense dataset**, available at [dhfbk/annotators-agreement-dataset](https://github.com/dhfbk/annotators-agreement-dataset).  
Please follow the link to obtain the dataset before running the scripts.

## Structure
- `scripts/` – preprocessing, embedding extraction, Mapper lens generation.  
- `analysis/` – computation of topological metrics (component purity, edge agreement, etc.).  
- `lens_outputs/` – results of Mapper lens computations.  
- `mapper/` – results of Mapper graph visualizations.  
- `data/` – placeholder for the MD-Offense dataset (not tracked in git).  
  - `raw_data/` – place MD-Offense raw files here.  
  - `processed_data/` – preprocessed data files.  
- `embeddings/` – base and fine-tuned embeddings (CLS, ignored by git).  
- `fine_tuned_models/` – checkpoints of fine-tuned models (ignored by git, can be regenerated with scripts).  
- `requirements.txt` – Python dependencies to reproduce experiments.  

## Reproducibility Note
Results may vary slightly due to randomness in model fine-tuning (e.g., seeds, initialization, or environment). This variance is expected, but the qualitative trends remain stable: fine-tuning yields modular, prediction-aligned regions, and the largest prediction–label gap appears in A0.

If you experiment with 2D lenses, we recommend using n_cubes=20 and overlap=0.2 (see Appendix for details). These hyperparameters worked well on MD-Offense, but may need adjustment for other datasets.


## Citation
If you use this code, please cite :
```bibtex
@inproceedings{rair-etal-2025-annotators,
    title = "When Annotators Disagree, Topology Explains: Mapper, a Topological Tool for Exploring Text Embedding Geometry and Ambiguity",
    author = "Rair, Nisrine  and
      Goupil, Alban  and
      Vrabie, Valeriu  and
      Chochoy, Emmanuel",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.426/",
    doi = "10.18653/v1/2025.emnlp-main.426",
    pages = "8468--8491",
    ISBN = "979-8-89176-332-6",
    abstract = "Language models are often evaluated with scalar metrics like accuracy, but such measures fail to capture how models internally represent ambiguity, especially when human annotators disagree. We propose a topological perspective to analyze how fine-tuned models encode ambiguity and more generally instances.Applied to RoBERTa-Large on the MD-Offense dataset, Mapper, a tool from topological data analysis, reveals that fine-tuning restructures embedding space into modular, non-convex regions aligned with model predictions, even for highly ambiguous cases. Over 98{\%} of connected components exhibit $\geq 90\%$ prediction purity, yet alignment with ground-truth labels drops in ambiguous data, surfacing a hidden tension between structural confidence and label uncertainty.Unlike traditional tool such as PCA or UMAP, Mapper captures this geometry directly uncovering decision regions, boundary collapses, and overconfident clusters. Our findings position Mapper as a powerful diagnostic tool for understanding how models resolve ambiguity. Beyond visualization, it also enables topological metrics that may inform proactive modeling strategies in subjective NLP tasks."
}

