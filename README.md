# MVTec Bottle Defect Classification

## **Overview**
This project is part of a **Machine Learning & AI internship assignment**.  
The goal is to build an image classification model to identify **defective** and **non-defective** bottles from the **MVTec Anomaly Detection** dataset.  
Additionally, the project extends into **multi-class classification** by identifying specific types of defects such as `broken_large`, `broken_small`, and `contamination`.

---

## **Project Structure**
mvtec-bottle-defect-classification/
├── data/ # Downloaded MVTec dataset
│ └── bottle/
│ ├── train/
│ └── test/
├── docs/ # Final report, screenshots, results
├── metadata/ # Class mappings, stats, weights
├── src/ # Source code
│ ├── generate_metadata.py
│ ├── prepare_dataset.py
│ ├── preprocessing.py
│ └── train_model.py
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── .gitignore # Files to ignore in Git
└── report.pdf # Summary of approach, results


---

## **Approach**

### 1. **Data Preparation**
- **Dataset**: MVTec AD - Bottle  
- **Classes**:
  - `good` (non-defective)
  - `broken_large`, `broken_small`, `contamination` (defective)
- **Preprocessing**:
  - Resized to **224x224**
  - Normalized using **ImageNet statistics**
  - Augmentations: `RandomRotation`, `RandomHorizontalFlip`, `ColorJitter`

---

### 2. **Data Splitting**
- Original dataset is already organized into:
  - `train/good/`: contains **only non-defective** images
  - `test/good/` and `test/<defect_type>/`: contains both **non-defective** and **defective** classes
- We used the `prepare_dataset.py` script to:
  - Load and label training and test data
  - Create **balanced dataloaders**
  - Ensure no data leakage between training and evaluation

---

### 3. **Handling Class Imbalance**
- **Class Weights**: Applied **log-scaled inverse frequency** weighting to reduce bias toward the dominant `good` class  
- **Sampler**: Used `WeightedRandomSampler` during training to ensure **balanced batches**

---

### 4. **Model Development**
- **Architecture**: `ResNet18` pretrained on ImageNet
- **Loss Function**: `CrossEntropyLoss` with computed class weights
- **Optimizer**: `Adam`
- **Scheduler**: `StepLR`
- **Framework**: PyTorch

---

### 5. **Evaluation Metrics**
- Accuracy  
- Precision, Recall, F1-Score (per class)  
- Confusion Matrix  

---

## **Performance**

| Class           | Precision | Recall | F1-Score |
|-----------------|-----------|--------|----------|
| Good            | 0.95      | 0.93   | 0.94     |
| Broken Small    | 0.78      | 0.82   | 0.80     |
| Broken Large    | 0.84      | 0.88   | 0.86     |
| Contamination   | 0.76      | 0.79   | 0.77     |
| **Macro Avg**   | **0.83**  | **0.86** | **0.84** |

> Earlier F1-scores for `contamination` and `broken_small` were below **0.60**.  
> Macro F1 average improved from ~**0.71** to **0.84**.

---

## **Key Highlights**
- **Log-scaled class weights** improved learning for underrepresented defect classes  
- **Data augmentation** helped boost recall for minority classes  
- **Efficient training** using ResNet18 without GPU acceleration

---

## **How to Clone**
```bash
git clone https://github.com/petkars/mvtec-bottle-defect-classification.git
cd mvtec-bottle-defect-classification
How to Run
1. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
2. Train the Model
bash
Copy
Edit
python src/train.py --epochs 25 --batch-size 32
3. Evaluate the Model
bash
Copy
Edit
python src/evaluate.py --weights saved_model.pth
Future Improvements
Try Vision Transformers (ViT) for spatial feature enhancement

Explore anomaly segmentation instead of classification

Experiment with semi-supervised learning to utilize unlabeled data

Credits
Dataset: MVTec AD

Architecture: PyTorch ResNet18

Contributor: Shubham Petkar