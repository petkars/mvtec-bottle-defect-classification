
# Dataset guide for ML & AI internship assignment

## 1. MVTec Anomaly Detection Dataset
- **Description**: A comprehensive dataset for benchmarking anomaly detection methods with industrial images.
- **Contents**: High-resolution images of 15 different object and texture categories, each with normal and defective samples. Defects include scratches, dents, contaminations, etc.
- **Link**: [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- **Notes**: Comes with pixel-precise ground truth annotations for the defects, which is helpful for both classification and segmentation tasks.

---

## 2. NEU Surface Defect Database
- **Description**: Specifically designed for surface defect detection on hot-rolled steel strips.
- **Contents**: 1,800 grayscale images classified into six defect types: rolled-in scale, patches, crazing, pitted surface, inclusion, and scratches.
- **Link**: [NEU Surface Defect Database](https://github.com/abin24/NEU_surface_defect_database)
- **Notes**: Useful for tasks focused on metal surface inspection.

---

## 3. DAGM Dataset
- **Description**: Provided by the German Association for Pattern Recognition for defect detection algorithms.
- **Contents**: Images of textured surfaces with various types of defects.
- **Link**: [DAGM Classification Dataset](https://hci.iwr.uni-heidelberg.de/node/6132)
- **Notes**: Contains both defective and non-defective images, suitable for binary classification.

---

## 4. Severstal Steel Defect Detection (Kaggle)
- **Description**: Dataset from a Kaggle competition aimed at detecting surface defects in steel manufacturing.
- **Contents**: Over 12,000 images with annotations for four defect classes.
- **Link**: [Severstal: Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection/data)
- **Notes**: Requires a Kaggle account to access. Comes with segmentation masks for defects.

---

## 5. Magnetic Tile Defect Dataset
- **Description**: Used for detecting defects in magnetic tiles, commonly found in industrial settings.
- **Contents**: Images categorized into defect types like blowhole, crack, fray, and uneven.
- **Link**: [Magnetic Tile Defect Dataset](https://github.com/abin24/magnetic-tile-defect-datasets)
- **Notes**: Provides a good mix of defect types for multi-class classification.

---

## 6. Kolektor Surface-Defect Dataset
- **Description**: A dataset for detecting defects on metal surfaces, provided by Kolektor Group.
- **Contents**: Images with annotated defects such as scratches and dents.
- **Link**: [KolektorSDD](https://www.vicos.si/resources/kolektorsdd/)
- **Notes**: Useful for binary classification and defect segmentation.

---

## 7. The Fabric Defect Dataset
- **Description**: Focused on detecting defects in textile fabrics, which can be analogous to certain industrial materials.
- **Contents**: Images with various fabric defects like holes, stains, and misweaves.
- **Link**: [Fabric Images Dataset](http://www.eng.tau.ac.il/~yaro/quality/assessing_quality.html)
- **Notes**: While textile-focused, methods can be transferable to other materials.

---

## 8. PCB Defect Detection
- **Description**: For detecting defects on Printed Circuit Boards (PCBs), relevant in electronics manufacturing.
- **Contents**: Images with defects such as missing components, shorts, and open circuits.
- **Link**: [PCB Dataset](https://github.com/Charmve/Surface-Defect-Detection)
- **Notes**: Good for exploring classification in a high-precision manufacturing context.

---

## If Existing Datasets Don't Fully Meet Your Needs

### 1. Data Augmentation
- Use techniques like **rotation**, **flipping**, and **noise addition** to artificially expand your dataset.

### 2. Synthetic Data Generation
- Tools like **Unity's Perception Package** can generate synthetic images with annotations, providing additional data for training.

### 3. Combine Datasets
- Merge multiple datasets to cover a broader range of equipment and defect types, increasing diversity in the data.

### 4. Collect Your Own Data
- If feasible, gather images directly from the specific industrial environment you're focusing on. This ensures the data is highly relevant to your problem.

---

## Next Steps

### 1. Download and Explore
- Choose a dataset and spend time understanding its **structure** and **contents** to ensure it meets your assignment goals.

### 2. Preprocessing
- Plan how you'll preprocess the dataset, including steps like:
  - **Resizing** to a consistent shape.
  - **Normalization** to standardize pixel values.
  - **Handling labels** to prepare for supervised learning.

### 3. Plan for Optional Objectives
- If you're aiming to:
  - **Classify defect types**, then
    1. Select datasets with **multi-class defect annotations** for classifying defect types (e.g., NEU Surface Defect Database).
    2. Prepare multi-class labels for defect types; balance imbalanced classes using oversampling or class weighting.
    3. Use pre-trained models (e.g., ResNet, EfficientNet) and fine-tune for your dataset.
    4. Evaluate with multi-class metrics (e.g., per-class precision/recall).
       
  - Or, **optimize for hardware acceleration** (e.g., GPU),
    1. Ensure dataset size and resolution are suitable for hardware-accelerated inference.
    2. Optimize data (e.g., resizing to 224x224 or 128x128, 8-bit quantization) for efficient edge device inference.
    3. Convert models using tools like **NVIDIA TensorRT** or **ONNX Runtime** for GPU/edge device deployment.
    4. Profile inference time and memory usage on target hardware. You can use tools like `docker` for this.
- Incorporate these objectives into your dataset selection and preprocessing pipeline.
