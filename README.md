# **ML4SCI GSoC 2025 Submissions**

This repository contains my submissions for the ML4SCI GSoC 2025 project, specifically addressing Tasks 1 and 2a of the [E2E Project](https://ml4sci.org/gsoc/2025/proposal_CMS1.html).

## **Tasks Overview**

### **Task 1**
**Datasets:**

https://cernbox.cern.ch/index.php/s/AtBT8y4MiQYFcgc (photons)
https://cernbox.cern.ch/index.php/s/FbXw3V4XNyYB3oA (electrons)

**Descriptions:**
- 32x32 matrices with two channels: hit energy and time for two types of particles, electrons and photons, hitting the detector.
- Use a ResNet-15 model to achieve highest possible classification score on the photons vs. electrons image dataset.
- Train the model on 80% of the data and evaluate on the remaining 20%.

### **Task 2a**
**Dataset:**

https://archive.ics.uci.edu/dataset/280/higgs

**Descriptions:**
- Train a Transformer Autoencoder model on the dataset above using only the first 21 features and only the first 1.1m events. The last 100k items are to be used as test set.
- Train a decoder which uses the latent space outputs of the Transformer encoder layer as inputs.
- Evaluate the performance of the classifier and present a ROC-AUC score for the final classifier.
- Original reference paper for a sense of good performance: https://arxiv.org/pdf/1402.4735.pdf
- Discuss choices made in model selection and optimization.

## Model Performance in Task 2a

The following table summarizes the performance of the models implemented on $900000$ training rows:

| Model                | Accuracy | AUC      |
|----------------------|----------|----------|
| Feedforward Decoder  | 0.737    | 0.81     |
| Transformer Decoder  | 0.747    | 0.824    |
| XGBoost              | 0.683    | 0.746    | 

*Note: Detailed results and analysis can be found in the respective notebook in the E2E folder.*