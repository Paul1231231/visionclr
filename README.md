VisionCLR: Self-Supervised Image Representation Learning with SimCLR
VisionCLR is a computer vision project that implements SimCLR (Simple Framework for Contrastive Learning of Visual Representations) with PyTorch. It learns useful image representations from unlabeled images using contrastive learning, then evaluates those representations through image retrieval and a nearest-neighbor classification baseline.
The project uses the Intel Image Classification dataset, which contains six scene categories: buildings, forest, glacier, mountain, sea, and street.
Current model configuration: ResNet-50 backbone, 2048-dimensional encoder features, and a two-layer projection head with 128-dimensional output embeddings. If you previously trained a ResNet-18 model, its checkpoint is not compatible with this ResNet-50 configuration; retrain before running retrieval/evaluation.

Features
- Two independent augmented views of each training image.
- ResNet-50 encoder and MLP projection head.
- NT-Xent (normalized temperature-scaled cross-entropy) contrastive loss.
- PyTorch training loop with AdamW optimizer.
- Apple Silicon MPS support, with CUDA/CPU fallbacks.
- Checkpoint and CSV training-log saving.
- Training-loss curve generation.
- Image retrieval using cosine similarity.
- Test-set evaluation using 1-nearest-neighbor classification.
How SimCLR works
For each source image, the data pipeline creates two randomly augmented views. Both views pass through the same encoder and projection head. NT-Xent encourages embeddings from the same source image to be similar while contrasting them against other images in the batch.
The encoder representation (h) is used for retrieval and downstream evaluation. The projection embedding (z) is used to calculate the contrastive training loss.
Image
  ├── Augmentation A ──┐
  └── Augmentation B ──┤
                       ▼
             Shared ResNet-50 encoder
                       ▼
             2048-dimensional h
                       ▼
               Projection MLP
                       ▼
              128-dimensional z
                       ▼
                  NT-Xent loss
                       ▼
             Backpropagation/update
Dataset
Dataset: Intel Image Classification on Kaggle
The archive commonly contains the following directories:
data/raw/
├── seg_train/
│   └── seg_train/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
└── seg_test/
    └── seg_test/
        ├── buildings/
        ├── forest/
        ├── glacier/
        ├── mountain/
        ├── sea/
        └── street/
Directory names can differ depending on how the archive is extracted. Pass the directory that directly contains the six class folders as --data-root, --train-root, or --test-root.
The training pipeline uses image augmentations and ignores labels when learning contrastive representations. The labels are used later for evaluation only.
Project structure
visionclr/
├── configs/
├── data/
│   └── raw/                   # Dataset files (not committed)
├── src/
│   └── visionclr/
│       ├── __init__.py
│       ├── data.py            # Datasets and image transforms
│       ├── models.py          # SimCLR encoder and projection head
│       ├── losses.py          # NT-Xent loss
│       ├── train.py           # Training entry point
│       └── retrieve.py        # Retrieval and 1-NN evaluation
├── scripts/
│   ├── check_data.py
│   ├── check_model.py
│   └── check_loss.py
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_loss.py
├── outputs/
│   └── simclr/                # Checkpoints, logs, and plots (not committed)
├── pyproject.toml
├── README.md
└── .gitignore
Some optional folders or test scripts may not exist yet in your local checkout.
Requirements
- Python 3.11 or 3.12 recommended
- PyTorch
- torchvision
- Pillow
- NumPy
- Matplotlib
- pytest (for tests)
Setup
Run these commands from the project root.
1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
2. Install dependencies
If the project has a pyproject.toml with dependencies configured:
pip install -e .
Otherwise, install the dependencies directly:
pip install torch torchvision pillow numpy matplotlib pytest
3. Download the dataset
One way is to use the Kaggle CLI. Configure your Kaggle API credentials according to Kaggle's instructions, then run:
pip install kaggle
mkdir -p data/raw
kaggle datasets download -d puneet6060/intel-image-classification \
  -p data/raw --unzip
Check the extracted layout:
find data/raw -type d -name buildings
find data/raw -type d -name sea
Run tests and component checks
The examples below assume the package is in src/ and commands are run from the project root.
PYTHONPATH=src pytest -q
Optional component checks, if the scripts exist in your checkout:
PYTHONPATH=src python scripts/check_data.py
PYTHONPATH=src python scripts/check_model.py
PYTHONPATH=src python scripts/check_loss.py
Train the model
Quick smoke test
Run a short integration test before training on the full dataset:
PYTHONPATH=src python -m visionclr.train \
  --epochs 1 \
  --batch-size 16 \
  --max-batches 10
This processes at most ten batches in one partial epoch. It verifies that the data loader, model, loss, gradients, and checkpoint-saving path work together.
Full training run
PYTHONPATH=src python -m visionclr.train \
  --epochs 5 \
  --batch-size 32
Useful arguments:
Argument	Purpose	Default
--data-root	Directory containing training class folders	data/raw/seg_train/seg_train
--epochs	Number of passes through the dataset	5
--batch-size	Number of source images per batch	32
--lr	AdamW learning rate	3e-4
--temperature	NT-Xent temperature	0.5
--max-batches	Maximum batches per epoch; 0 means all	0
--num-workers	DataLoader worker processes	0
--output-dir	Directory for checkpoints/logs/plots	outputs/simclr


The script selects MPS when available, then CUDA, otherwise CPU. If you encounter memory pressure, try reducing the batch size.
Training outputs
The default output directory is:
outputs/simclr/
├── simclr_checkpoint.pt
├── training_log.csv
└── loss_curve.png
- simclr_checkpoint.pt: model state, optimizer state, epoch, loss, and configuration metadata.
- training_log.csv: average loss and number of batches per epoch.
- loss_curve.png: plot of average training loss across epochs.
A decreasing loss can be encouraging, but it does not by itself prove that the learned features are useful. Evaluate the representations on held-out data as well.
Use the model: image retrieval
Once training has completed, the retrieval script can embed images and retrieve the closest training-set images by cosine similarity.
Retrieve images similar to one query image
Replace the example filename with a real image path:
PYTHONPATH=src python -m visionclr.retrieve \
  --image "data/raw/seg_test/seg_test/mountain/your_image.jpg" \
  --top-k 5
The script prints the top-k most similar training images, their dataset labels, and cosine similarity scores. Similarity is not a calibrated class probability.
Evaluate the labeled test set
PYTHONPATH=src python -m visionclr.retrieve --evaluate-test
This performs 1-nearest-neighbor evaluation: each test image is assigned the label of the most similar training-image embedding. The script reports overall accuracy and per-class accuracy.
You can combine retrieval and evaluation:
PYTHONPATH=src python -m visionclr.retrieve \
  --image "data/raw/seg_test/seg_test/sea/your_image.jpg" \
  --top-k 5 \
  --evaluate-test
The train and test class-folder mappings must match. Keep the test set out of model training and use it only for evaluation.
Important notes
1. Architecture and checkpoint compatibility: ResNet-18 produces 512-dimensional encoder features; ResNet-50 produces 2048-dimensional features. A checkpoint trained with one architecture cannot be loaded into the other without an explicit weight-conversion strategy. Retrain after changing the backbone.
2. Batch size matters: SimCLR uses other views in the same batch as negative examples. Larger batches can provide more negatives but require more memory.
3. Labels are not used for contrastive training: The class labels are ignored by the contrastive dataset during training.
4. Evaluation is a baseline: 1-nearest-neighbor accuracy measures whether similar embeddings tend to share labels. It is not the same as training a supervised classifier.
5. Avoid committing large artifacts: Keep raw datasets, virtual environments, caches, and checkpoints out of Git unless there is a deliberate reason to publish them.
Suggested .gitignore
# Python
__pycache__/
*.py[cod]
.venv/
.pytest_cache/

# Environment variables and secrets
.env
.env.*

# Dataset and generated artifacts
data/raw/
data/processed/
outputs/
*.pt
*.pth

# OS/editor files
.DS_Store
.vscode/
.idea/
Roadmap
- [x] Build the image data pipeline with contrastive augmentations.
- [x] Implement the SimCLR model and NT-Xent loss.
- [x] Train the model and save checkpoints/logs.
- [x] Retrieve similar images using learned representations.
- [x] Evaluate representations with a 1-nearest-neighbor baseline.
- [ ] Add a linear-evaluation protocol with a frozen encoder and trained linear classifier.
- [ ] Compare ResNet-18 and ResNet-50 using consistent training settings.
- [ ] Track experimental configurations and compare results systematically.
Learning objectives
This project demonstrates practical understanding of:
- Self-supervised representation learning
- Contrastive learning and NT-Xent loss
- CNN encoders and projection heads
- PyTorch training loops and gradient-based optimization
- Data augmentation and train/test separation
- Feature extraction, cosine similarity, and nearest-neighbor retrieval
- Experiment logging and reproducibility
References
- SimCLR paper: A Simple Framework for Contrastive Learning of Visual Representations
- PyTorch documentation
- torchvision ResNet models
- Intel Image Classification dataset
