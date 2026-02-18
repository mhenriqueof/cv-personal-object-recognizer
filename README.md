# **Personal Object Recognizer**
A real-time few-shot computer vision system that learns and recognizes personal objects from up to 4
images. Built with NumPy, OpenCV, YOLO26 and MobileNetV3.


## **Objective**
The goal of this project was to build a system that integrates Computer Vision, Deep Learning
and Linear Algebra, fields I have recently studied, to apply and consolidate the knowledge I
acquired.

Beyond the main ML, the project was also an opportunity to practice software engineering principles,
including code versioning, structured documentation, modular design, clean code practices, performance
optimization and reproducibility.


## **Overview**
Instead of retraining a classifier, the system utilizes:

* Pretrained object detection (YOLO)
* Feature embeddings (MobileNet)
* Prototype-based representation
* Cosine similarity for matching

New objects can be added instantly without retraining, making the system lightweight and incremental.


## **How It Works**
The application operates in two modes:

### 1. Recognize Mode (Default)
Real-time multi-object detection and recognition.

Pipeline:

```
Frame → YOLO Detection → Crop Objects → Batch Embedding Extraction → Cosine Similarity → Best Match → Display Results
```

Features:

* Up to 3 simultaneous objects
* Color confidence levels
* Live FPS monitoring

### 2. Register Mode
Learns a new object from a maximum of 4 images.

Pipeline:

```
Capture Images → Brightness Augmentation → Embedding Extraction → Prototype Averaging → JSON Storage
```

No retraining required, learning is instantaneous.


## **Core Concepts**

### Feature Extraction
* **Model**: MobileNetV3-Small
* Classification head removed
* Global average pooling added
* Output: **576-dimensional L2-normalized embedding**

### Prototype Representation
**Each object is represented by a single vector:**

$$
prototype = \frac{1}{n} \sum_{i=1}^{n} embedding_i
$$

The averaged vector is re-normalized before storage.

### Similarity Matching
Object recognition is performed using **cosine similarity** between the query embedding and each
stored prototype.

**Cosine similarity is defined as:**

$$
\text{sim}(q, p) = \frac{q \cdot p}{|q| |p|}
$$

**Since both vectors are L2-normalized:**

$$
|q| = |p| = 1
$$

**The expression simplifies to dot product:**

$$
\text{sim}(q, p) = q \cdot p
$$

**Recognition is performed by selecting:**

$$
\hat{y} = \arg\max_{p_i} (q \cdot p_i)
$$

* where ( $p_i$ ) represents each stored object prototype.

Predictions are accepted or rejected based on configurable confidence thresholds in `configs/config.yaml`.


## **Performance Optimization**

Significant attention was given to optimization and system efficiency.

### Optimizations Implemented

1. **Batch embedding extraction** <br>
   Single forward pass for multiple object crops.

2. **Frame skipping with caching** <br>
   YOLO runs every 10 frames.

3. **Prototypes/labels caching** <br>
   Prevents unnecessary database reloads.

4. **Deterministic CUDA configuration** <br>
   Fixed seeds `cudnn.deterministic = True` (ensures reproducibility).


### Results
**Hardware:** i3-6100, GTX 1060 3GB, 8GB RAM <br>
**Webcam:** 640×480 @ 30 FPS

#### **Before Optimization**
| Scenario | FPS (average) | Notes |
|----------|-----|-------|
| No objects | 30 | Webcam max |
| 1 object | 21 | Sequential extraction |
| 2 objects | 15 | Linear scaling bottleneck |

#### **After Optimization**
| Scenario | FPS (average) | Improvement |
|----------|-----|-------------|
| No objects | 30 | – |
| 1 object | 24 | **+14%** |
| 2 objects | 24 | **+60%** |


## **Usage**
Detection depends on [YOLO's pretrained classes](https://docs.ultralytics.com/datasets/detect/coco/#dataset-yaml).

### Controls
| Key     | Mode      | Action              |
| ------- | --------- | ------------------- |
| `R`     | Recognize | Enter Register mode |
| `C`     | Recognize | Clear database      |
| `Q`     | Both      | Quit / Return       |
| `Space` | Register  | Capture image       |
| `F`     | Register  | Finish registration |

### 1. Recognize Objects
```bash
# Green → Confident (>0.8)
# Yellow → Uncertain  (0.7-0.8)  
# Gray → Unknown (<0.7)
```

### 2. Register a New Object
```bash
# Press 'R' in Recognize mode
# Capture a maximum of 4 images
# Press 'F' to finish registration
```

### **Demonstration**
[Demo Video on LinkedIn](https://www.linkedin.com/posts/mhenriqueof_computervision-deeplearning-linearalgebra-ugcPost-7428051253152092160-f6Ll?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE7A8jkBQPtwJJnf3HYdYbNZBwBCyxS8Xg0)


## **Installation**
### Requirements
- Python 3.13

### Clone the Repository
```bash
git clone https://github.com/mhenriqueof/cv-personal-object-recognizer.git
cd cv-personal-object-recognizer
```

### Install PyTorch with CUDA (~3GB)
```bash
pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Install Remaining Dependencies
```bash
pip install -r requirements.txt
```

On first run, the system automatically downloads:

* YOLO26 Nano weights
* MobileNetV3-Small


## **Project Structure**
```
cv_personal_object_recognizer/
├── configs/ # YAML configuration
├── src/ # Source code
│   ├── core/ # Core components
│   │   ├── camera.py    # OpenCV camera handler
│   │   ├── detector.py  # YOLO object detection
│   │   ├── extractor.py # MobileNetV3 feature extraction
│   │   └── memory.py    # Database manager
│   ├── utils/ # Utilities
│   │   ├── augmentation.py # Brightness augmentation
│   │   ├── config.py       # YAML config loader
│   │   ├── fps_tracker.py  # Performance monitoring
│   │   ├── input_helper.py # Input validation for object name
│   │   ├── logger.py       # Logging setup
│   │   ├── seed.py         # Reproducibility seeds
│   │   └── system_mode.py  # Enum for system modes
│   ├── app.py        # Main application orchestrator
│   ├── register.py   # Registration workflow
│   └── recognizer.py # Recognition workflow
├── models/ # YOLO weights (downloaded automatically)
├── data/   # Database and captured images
└── main.py # Entry point
```

---

Even though this type of system (unimodal) will not be my focus in the future, I am very happy to have
put what learned into practice and to have learned so much once again during the development process.

Thanks!
