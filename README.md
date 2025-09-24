# Fine-Tuned Language Models – Research Repository

This repository documents a series of fine-tuning experiments conducted on different pretrained Transformer-based models. The aim is to systematically evaluate the performance of various architectures and fine-tuning techniques on benchmark NLP tasks. Each notebook in the repository corresponds to one experiment, detailing the model, dataset, evaluation, and results.

The overarching goal is to build a comparative reference for fine-tuning strategies, parameter-efficient training, and model behavior across tasks.

---

## Model 1: DistilBERT – Sentiment140

* **Model**: [DistilBERT](https://arxiv.org/abs/1910.01108) (Sanh et al., 2019)
* **Base Checkpoint**: `distilbert-base-uncased`
* **Dataset**: [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) (1.6M tweets, binary sentiment classification)
* **Fine-Tuning Method**: Full fine-tuning
* **Training Setup**: 3 epochs, global steps = 6564
* **Training Loss**: 0.1729
* **Evaluation Metrics** (test set, 60,000 samples):

  * Accuracy: **83.86%**
  * Precision: 0.84
  * Recall: 0.84
  * F1-score: 0.84
* **Runtime**: 2688s (\~44.8 min), 156 samples/sec
* **Notebook**: [Fine-Tuning DistilBERT on Sentiment140](notebooks/sentiment140-distilbert-finetune-ipynb.ipynb)

---

## Model 2: BERT-base – (Upcoming)

* **Model**: [BERT](https://arxiv.org/abs/1810.04805) (Devlin et al., 2018)
* **Base Checkpoint**: `bert-base-uncased`
* **Dataset**: Sentiment140
* **Fine-Tuning Method**: Planned full fine-tuning
* **Status**: Pending experiment

---

## Future Directions

This catalog will expand to include:

* **RoBERTa**, **DeBERTa**, and other Transformer variants.
* **Parameter-efficient fine-tuning (PEFT)** techniques:

  * LoRA
  * Prefix-Tuning
  * BitFit
* Cross-dataset comparisons (IMDb, SST-2, Amazon Reviews, etc.).

---

## References

1. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
3. Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
4. Li, X. L., & Liang, P. (2021). *Prefix-Tuning: Optimizing Continuous Prompts for Generation*. [arXiv:2101.00190](https://arxiv.org/abs/2101.00190)
5. Zaken, E. B., Ravfogel, S., & Goldberg, Y. (2021). *BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models*. [arXiv:2106.10199](https://arxiv.org/abs/2106.10199)
