# TDA-NLP-Representations

*When Annotators Disagree, Topology Explains: Mapper, a Topological Tool for Exploring Text Embedding Geometry and Ambiguity*  (accepted at **EMNLP 2025, Main Conference**).  

We study how fine-tuned encoder-only models represent ambiguous text data.  
Using **Mapper**, a tool from topological data analysis, we show that fine-tuning reshapes embedding space into modular, prediction-aligned regions, even when annotators disagree.  
Our analysis highlights a persistent gap between model confidence and human label ambiguity, demonstrating Mapper’s value as an exploratory tool for representation geometry.

For full details, see the paper ( PDF coming soon : ACL Anthology link will be added after publication)


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
  - `processed/` – preprocessed data files.  
- `embeddings/` – base and fine-tuned embeddings (CLS and mean; ignored by git).  
- `fine_tuned_models/` – checkpoints of fine-tuned models (ignored by git, can be regenerated with scripts).  
- `requirements.txt` – Python dependencies to reproduce experiments.  


## Citation
If you use this code, please cite (BibTeX coming soon).

