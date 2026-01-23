# Image Captioning with Attention Mechanism

Deep learning model for automatic image caption generation using CNN-LSTM architecture with soft attention.

## Overview

This project implements an encoder-decoder architecture that generates natural language descriptions of images. The encoder uses a pretrained ResNet-50 CNN to extract visual features, while the decoder employs an LSTM with attention mechanism to generate captions word by word, focusing on relevant image regions at each timestep.

## Motivation

Automatic image captioning bridges computer vision and natural language processing, enabling applications in accessibility (describing images for visually impaired users), content organization, and image retrieval. This implementation demonstrates the effectiveness of attention mechanisms in generating contextually relevant descriptions.

## Repository Structure

```
.
├── data_and_training.ipynb    # Main training notebook
├── inference.ipynb             # Caption generation on test images
├── caption_data/               # Dataset directory
│   ├── Images/                 # Image files
│   ├── captions.txt           # Image-caption pairs
│   └── test/                  # Test images
├── best_model.pt              # Trained model checkpoint
├── vocab.json                 # Vocabulary mappings
├── training_history.json      # Training metrics
├── training_history.png       # Loss curves
├── data_exploration.png       # Dataset statistics
├── sample_predictions.png     # Sample outputs
└── bleu_scores.json          # Evaluation metrics
```

## Requirements

### Dependencies

- Python 3.8+
- PyTorch 2.0+
- torchvision
- Pillow
- NumPy
- Matplotlib
- tqdm
- NLTK (optional, for BLEU evaluation)

### Hardware

- CPU: Works on CPU (slower training)
- GPU: CUDA-compatible GPU recommended for faster training
- RAM: Minimum 8GB, 16GB recommended
- Storage: 2GB for dataset, 400MB for model checkpoint

## Installation

1. Clone or download this repository:
```bash
git clone <repository-url>
cd DeepLearning
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install torch torchvision pillow numpy matplotlib tqdm nltk
```

4. (Optional) Download NLTK data for BLEU evaluation:
```python
python -c "import nltk; nltk.download('punkt')"
```

## Dataset

This project uses the Flickr8k dataset (or similar image-caption datasets). The dataset should be structured as:

- `caption_data/Images/`: Directory containing image files
- `caption_data/captions.txt`: Text file with format `image_name,caption`

Each image should have multiple reference captions (typically 5 captions per image).

## Model Architecture

### Encoder
- **Base Model**: ResNet-50 pretrained on ImageNet
- **Output**: 7×7×2048 feature map (49 spatial locations, 2048 features each)
- **Fine-tuning**: Encoder frozen for first 5 epochs, then fine-tuned

### Attention Mechanism
- **Type**: Soft attention (Bahdanau attention)
- **Function**: Computes weighted combination of encoder features at each decoding step
- **Regularization**: Doubly stochastic attention (attention weights sum to 1)

### Decoder
- **Architecture**: Single-layer LSTM
- **Embedding**: 256-dimensional word embeddings
- **Hidden State**: 768 dimensions
- **Attention**: 512 dimensions
- **Vocabulary**: Built from training captions (minimum frequency threshold: 5)

## Training

### Configuration

Default hyperparameters in `data_and_training.ipynb`:

```python
EMBED_DIM = 256          # Word embedding dimension
HIDDEN_DIM = 768         # LSTM hidden state dimension
ATTENTION_DIM = 512      # Attention mechanism dimension
BATCH_SIZE = 64          # Training batch size
NUM_EPOCHS = 25          # Total training epochs
LEARNING_RATE = 3e-4     # Initial learning rate
ENCODER_LR = 1e-4        # Encoder learning rate (after unfreezing)
MIN_WORD_FREQ = 5        # Minimum word frequency for vocabulary
```

### Training Procedure

1. Open `data_and_training.ipynb` in Jupyter Notebook or JupyterLab

2. Run cells sequentially:
   - Cells 0-1: Import libraries and set configuration
   - Cells 3-5: Define model components
   - Cells 7-9: Load and process dataset
   - Cells 11-13: Initialize model architecture
   - Cells 15-17: Execute training loop

3. Training stages:
   - **Stage 1 (Epochs 1-5)**: Train decoder only, encoder frozen
   - **Stage 2 (Epochs 6-25)**: Fine-tune entire model

4. The model automatically saves the best checkpoint based on validation loss to `best_model.pt`

### Loss Functions

- **Cross-Entropy Loss**: Measures caption prediction accuracy
- **Attention Regularization**: Encourages attention to cover all image regions

Training displays two loss values:
- `Train Loss (CE)`: Comparable to validation loss
- `Train Loss (Att Reg)`: Attention penalty (training only)

## Inference

To generate captions for new images:

1. Open `inference.ipynb`

2. The notebook loads the trained model from `best_model.pt`

3. Place test images in `caption_data/test/` or specify image paths

4. Run inference cells to generate captions with beam search (beam size: 3)

5. Results include generated captions and attention visualizations

## Evaluation

Model performance is evaluated using BLEU scores (Bilingual Evaluation Understudy):

- **BLEU-1**: Unigram precision
- **BLEU-2**: Bigram precision
- **BLEU-3**: Trigram precision
- **BLEU-4**: 4-gram precision (most commonly reported)

Run evaluation cells in `data_and_training.ipynb` after training completes. Results are saved to `bleu_scores.json` and `bleu_scores.png`.

### Expected Performance

Typical BLEU-4 scores on Flickr8k:
- Baseline (no attention): 0.20-0.25
- With attention: 0.25-0.30
- State-of-the-art: 0.30-0.35

Note: Results vary based on dataset size, training duration, and hyperparameters.

## Results

Training outputs several visualization files:

- `training_history.png`: Training and validation loss curves
- `data_exploration.png`: Dataset statistics and vocabulary analysis
- `sample_predictions.png`: Generated captions on validation images
- `bleu_scores.png`: BLEU score bar chart

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**:
- Reduce `BATCH_SIZE` in configuration (try 32 or 16)
- Decrease `HIDDEN_DIM` or `ATTENTION_DIM`
- Use CPU instead of GPU

**Poor Caption Quality**:
- Train for more epochs (25+ recommended)
- Increase dataset size
- Adjust `MIN_WORD_FREQ` to include more vocabulary

**Slow Training**:
- Enable GPU acceleration
- Increase `BATCH_SIZE` if memory allows
- Reduce `NUM_EPOCHS` for testing

**Import Errors**:
- Ensure all dependencies are installed: `pip list`
- Activate virtual environment
- Reinstall packages: `pip install --upgrade torch torchvision`

## Technical Details

### Vocabulary Processing
- Captions are lowercased
- Punctuation and numbers removed
- Words appearing less than `MIN_WORD_FREQ` times mapped to `<UNK>`
- Special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`

### Data Augmentation
- Random resizing and cropping
- Random horizontal flip
- Color jittering (brightness, contrast, saturation)

### Optimization
- Optimizer: Adam
- Learning rate scheduling: ReduceLROnPlateau (factor=0.5, patience=3)
- Gradient clipping: Max norm 5.0
- Teacher forcing during training

## References

### Model Architecture
- Show, Attend and Tell: Neural Image Caption Generation with Visual Attention (Xu et al., 2015)
- Deep Residual Learning for Image Recognition (He et al., 2016)

### Documentation Best Practices
- [GitHub README Template for ML Projects](https://github.com/catiaspsilva/README-template)
- [Deep Learning Project Template](https://github.com/L1aoXingyu/Deep-Learning-Project-Template/blob/master/README.md)
- [README Best Practices - Tilburg Science Hub](https://tilburgsciencehub.com/building-blocks/store-and-document-your-data/document-data/readme-best-practices/)

## License

This project is provided for educational purposes. Please respect the licenses of the datasets and pretrained models used.

## Acknowledgments

- ResNet-50 pretrained weights from PyTorch model zoo
- Dataset: Flickr8k (or equivalent image-caption dataset)
- Attention mechanism implementation based on "Show, Attend and Tell" paper
