# BitDance Setup Guide 🚀

Complete setup instructions for running the BitDance policy violation detection system with real AI models.

## Overview

BitDance is a restaurant review analysis system that uses machine learning to detect policy violations. The system consists of:
- **Frontend**: React dashboard for testing and visualization
- **Backend**: Flask API for model inference
- **ML Pipeline**: Training pipeline for policy violation detection models

## Prerequisites

- Python 3.8+ 
- Node.js 16+
- Git

## Quick Start (3 Terminal Setup)

### Terminal 1: Model Training (Run First)

```bash
# 1. Clone and navigate to project
git clone <repository-url>
cd bitdance

# 2. Install Python dependencies
pip install -r requirements.txt
pip install datasets transformers torch

# 3. Set up data pipeline (one-time setup)
python -m src.orchestrator.cli run-task ingest_all
python -m src.orchestrator.cli run-task normalize

# 4. Train the model (takes ~10-20 minutes)
python -m src.orchestrator.cli run-task train
```

**⏱️ Training Time:** ~10-20 minutes for DistilBERT model

### Terminal 2: Backend Server

```bash
# After model training completes, start the backend
cd bitdance
python backend/server.py
```

**✅ Expected Output:**
```
* Running on http://127.0.0.1:5000
* Model inference endpoint available at /api/inference/predict
```

### Terminal 3: Frontend Development Server

```bash
# Start the React frontend
cd bitdance/frontend
npm install
npm run dev
```

**✅ Expected Output:**
```
Local:   http://localhost:5173/
```

## Detailed Setup Steps

### 1. Data Pipeline Setup (First Time Only)

The system needs to process your restaurant review data through several stages:

```bash
# Check available tasks
python -m src.orchestrator.cli list

# Process data pipeline
python -m src.orchestrator.cli run-task ingest_all    # Converts CSV to parquet
python -m src.orchestrator.cli run-task normalize    # Prepares training data
```

**Data Flow:**
```
data/reviews.csv → data/interim/ → data/annotated/ → data/processed/
```

### 2. Model Training Details

The training process uses your annotated restaurant review data to train a DistilBERT model for policy violation detection:

```bash
python -m src.orchestrator.cli run-task train
```

**Training Configuration:**
- Model: `distilbert-base-uncased`
- Labels: `irrelevant_content`, `advertisement`, `review_without_visit`
- Batch size: 16
- Learning rate: 5e-5
- Epochs: 3

**Expected Output Directory:**
```
models/
└── encoder/
    └── enc-distilbert-base-uncased/
        ├── config.json
        ├── pytorch_model.bin
        └── tokenizer files...
```

### 3. API Testing

Once the backend is running, you can test the model inference:

```bash
curl -X POST http://localhost:5000/api/inference/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "encoder/enc-distilbert-base-uncased",
    "family": "encoder",
    "text": "Great food and service!",
    "threshold": 0.5
  }'
```

**Expected Response:**
```json
{
  "model": "encoder/enc-distilbert-base-uncased",
  "family": "encoder",
  "labels": ["irrelevant_content", "advertisement", "review_without_visit"],
  "threshold": 0.5,
  "n": 1,
  "predictions": [
    {
      "probs": [0.1, 0.05, 0.02],
      "pred": [0, 0, 0]
    }
  ]
}
```

## Frontend Features

### Policy Violations Dashboard

Navigate to `http://localhost:5173` and go to the "Policy Violations" tab to:

- **Test Real AI Analysis**: Input restaurant review text and get live predictions
- **View Violation Categories**: See breakdown of different policy violation types
- **Risk Assessment**: Get confidence scores and risk levels
- **Model Transparency**: See which trained model is being used

### Demo vs Real Mode

- **🟢 Real Mode**: Shows when connected to trained models
- **🟡 Demo Mode**: Fallback keyword-based analysis when models aren't available

## Troubleshooting

### Common Issues

**1. Training Fails - Missing Data**
```bash
# Error: FileNotFoundError: data/interim/restaurant_reviews/reviews.parquet
# Solution: Run data pipeline first
python -m src.orchestrator.cli run-task ingest_all
```

**2. Backend 400 Errors**
```bash
# Error: Model path not found
# Solution: Check trained model exists
ls models/encoder/
```

**3. Parquet File Corruption**
```bash
# Error: Repetition level histogram size mismatch
# Solution: Delete and recreate parquet files
rm data/annotated/restaurant_reviews/annotations.parquet
python -c "import pandas as pd; df = pd.read_csv('data/annotated/restaurant_reviews/annotations.csv'); df.to_parquet('data/annotated/restaurant_reviews/annotations.parquet', index=False)"
```

**4. Missing Dependencies**
```bash
# Install missing ML libraries
pip install datasets transformers torch
pip install pandas pyarrow

# Install frontend dependencies
cd frontend && npm install
```

### Data Requirements

Your `data/reviews.csv` should have columns for:
- Review text
- Policy violation labels
- Categories (advertisement, irrelevant_content, etc.)

### Model Configuration

Edit `configs/base.yaml` to customize:
- Model architecture (`model_name`)
- Training parameters (`lr`, `epochs`, `batch_size`)
- Data paths and preprocessing options

## Development Workflow

### Adding New Models

1. Update `configs/base.yaml` with new model configuration:
```yaml
models:
  - family: "encoder"
    model_name: "bert-large-cased"
    batch_size: 8
    lr: 3e-5
    epochs: 5
    run_name: "enc-bert-large-cased"
```

2. Train the new model:
```bash
python -m src.orchestrator.cli run-task train
```

3. Update frontend API calls to use new model path

### Testing Different Models

The system supports multiple model families:
- **encoder**: BERT, DistilBERT, RoBERTa
- **sft**: Fine-tuned language models (Gemma, Mistral)
- **baseline**: Traditional ML approaches

### Production Deployment

For production use:
1. Train models on full dataset
2. Use GPU acceleration for faster training
3. Configure proper environment variables
4. Set up model versioning and monitoring

## Performance Expectations

| Component | Startup Time | Resource Usage |
|-----------|--------------|----------------|
| Model Training | 10-60 minutes | High CPU/GPU |
| Backend Server | 10-30 seconds | Medium CPU |
| Frontend | 5-10 seconds | Low |

## Next Steps

Once everything is running:
1. Test the policy violation detection with your restaurant reviews
2. Experiment with different model architectures
3. Analyze model performance on your specific data
4. Customize the dashboard for your use case

## Support

- Check terminal outputs for detailed error messages
- Review model training logs in the console
- Test API endpoints individually if frontend issues occur
- Verify all file paths match your data structure

---

**Happy coding! 🎉** Your policy violation detection system should now be running with real AI models.
