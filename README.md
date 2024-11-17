# **Dimensionality Reduction and Drift Detection with Alibi-Detect**

---

## **Project Description**

This project demonstrates the use of an **autoencoder** for dimensionality reduction applied to language model embeddings (**BERT**) and the implementation of **data drift detection** using the **Alibi-Detect** package.  
It also includes an analysis of bias in the data, showing how imbalanced distributions can be detected as drift.

---

## **Main Features**

1. **Dimensionality Reduction with Autoencoder:**
   - Reduction of BERT embeddings (768 dimensions) to a compact latent space (64 dimensions).
   - PyTorch is used to build and train the autoencoder.

2. **Drift Detection with Alibi-Detect:**
   - Drift detection based on the Kolmogorov-Smirnov test (**KSDrift**).
   - Monitoring of distributional shifts between reference and test samples.

3. **Bias Analysis in Data:**
   - Comparison between balanced and biased datasets.
   - Evaluation of the impact of bias on drift detection.

---

## **Requirements**

### **Required Python Libraries:**
- `torch`: For implementing the autoencoder.
- `alibi_detect`: For drift detection.
- `numpy`: For data manipulation.
- `scikit-learn`: For additional tools, such as data splitting.

To install the dependencies, run:
```bash
pip install torch alibi-detect numpy scikit-learn
```

---

## **Code Structure**

### **1. Autoencoder: Dimensionality Reduction**
- Defines an **autoencoder** with two parts:
  - **Encoder**: Reduces the dimension of the embeddings to 64.
  - **Decoder**: Reconstructs the original data for training.
- Trained on a sample of BERT embeddings to minimize reconstruction error.

### **2. Data Encoding**
- The embedding data (`sampled_embeddings`) is passed through the encoder to obtain a reduced, latent space representation.

### **3. Drift Detection**
- A **KSDrift** detector is created with a reference sample.
- Drift detection is performed between the reference sample and new incoming samples, which may be balanced or biased.

### **4. Bias Analysis**
- Creation of a biased sample (90% class 0, 10% class 1).
- Comparison of drift detection results between:
  - A balanced sample.
  - A biased sample.

---

## **Project Files**

| File                  | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `autoencoder.py`       | Autoencoder implementation (model, training, and encoding).                |
| `drift_detection.py`   | Drift detection pipeline using Alibi-Detect's KSDrift.                      |
| `data_processing.py`   | Data preparation and manipulation (balanced and biased sampling).          |
| `run_experiment.py`    | Script to execute the full experiment (training, encoding, drift detection).|

---

## **How to Run the Code**

### **Step 1: Data Preparation**
Before running the experiments, ensure that the input data (e.g., BERT embeddings) is preprocessed and ready. You can use your own dataset or simulate one.

### **Step 2: Train the Autoencoder**
Run the autoencoder training by executing:
```bash
python autoencoder.py
```
This will train the model to learn the encoding and decoding process on the BERT embeddings.

### **Step 3: Drift Detection**
To detect drift, execute the drift detection pipeline:
```bash
python drift_detection.py
```
The script will perform drift detection using the KSDrift detector on the provided reference and test datasets.

### **Step 4: Run Experiment**
To run the full experiment, which includes training the autoencoder and performing drift detection on both balanced and biased datasets, run:
```bash
python run_experiment.py
```

---

## **Example Output**

When running the drift detection pipeline, you will see outputs indicating whether drift has been detected (True or False) along with any statistical details from the KS test. 

---

## **Future Improvements**

- Implement other drift detection methods such as **ClassifierDrift** and **PredictionDrift**.
- Extend the autoencoder model with additional layers or advanced architectures like Variational Autoencoders (VAEs).
- Explore real-world datasets with more complex biases and drift scenarios.

