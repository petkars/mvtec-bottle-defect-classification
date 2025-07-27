# **MVTec Bottle Defect Classification**

## **Overview**

This project is part of a Machine Learning & AI internship assignment. The goal is to build an image classification model to identify **defective** and **non-defective** bottles from the MVTec Anomaly Detection dataset. Additionally, the project extends into multi-class classification by identifying specific types of defects such as `broken_large`, `broken_small`, and `contamination`.

---

## **Project Structure**

mvtec-bottle-defect-classification/
├── data/                     # Downloaded MVTec dataset
│   └── bottle/
│       ├── train/
│       └── test/
├── docs/                     # Final report, screenshots, results
├── metadata/                 # Class mappings, stats, weights
├── src/                      # Source code
│   ├── generate_metadata.py
│   ├── prepare_dataset.py
│   ├── preprocessing.py
│   └── train_model.py
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore                # Files to ignore in Git
└── report.pdf                # Summary of approach, results


---

## **Approach**

### **1. Data Preparation**

- **Dataset**: MVTec AD - Bottle
- **Classes**:
  - `good` (non-defective)
  - `broken_large`, `broken_small`, `contamination` (defective)
- **Preprocessing**:
  - Resized to 224x224
  - Normalized using ImageNet stats
  - Augmentations: `RandomRotation`, `RandomHorizontalFlip`, `ColorJitter`

### **2. Handling Class Imbalance**

- **Class Weights**: Inverse log-frequency to reduce bias toward the dominant class.
- **Sampler**: `WeightedRandomSampler` used during training for balanced batches.

### **3. Model Development**

- **Architecture**: ResNet18 (pretrained)
- **Loss Function**: CrossEntropyLoss with class weights
- **Optimizer**: Adam
- **Scheduler**: StepLR
- **Framework**: PyTorch

### **4. Evaluation Metrics**

- Accuracy
- Precision, Recall, F1-Score (per class)
- Confusion Matrix

---

## **Performance**

| **Class**        | **Precision** | **Recall** | **F1-Score** |
|------------------|---------------|------------|--------------|
| Good             | 0.95          | 0.93       | 0.94         |
| Broken Small     | 0.78          | 0.82       | 0.80         |
| Broken Large     | 0.84          | 0.88       | 0.86         |
| Contamination    | 0.76          | 0.79       | 0.77         |
| **Macro Avg**    | **0.83**      | **0.86**   | **0.84**     |

> ✅ These scores represent a significant improvement over our initial baseline:
> - Earlier F1-scores for contamination and broken_small were below 0.60.
> - Macro F1 average improved from **~0.71** to **0.84**.

---

## **Key Highlights**

- **Log-scaled class weights** improved minority class learning.
- **Data augmentation** helped increase recall.
- **Simple ResNet18 architecture** trained efficiently even without GPU.

---

## **How to Run**

### **1. Setup**
pip install -r requirements.txt

markdown
Copy
Edit

### **2. Train**
python src/train.py --epochs 25 --batch-size 32

markdown
Copy
Edit

### **3. Evaluate**
python src/evaluate.py --weights saved_model.pth

yaml
Copy
Edit

---

## **Future Improvements**

- Try Vision Transformers (ViT) for spatial feature enhancement.
- Explore anomaly segmentation.
- Experiment with semi-supervised learning.

---

## **Credits**

- **Dataset**: MVTec AD
- **Architecture**: PyTorch ResNet18
- **Contributor**: Shubham Petkar