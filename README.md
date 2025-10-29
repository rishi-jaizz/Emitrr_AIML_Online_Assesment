# Medical Transcription NLP Pipeline - Emitrr

A comprehensive AI system for medical transcription, NLP-based summarization, and sentiment analysis designed for physician-patient conversations.

## Overview

This project implements an end-to-end NLP pipeline that processes medical transcripts to:
- Extract medical entities (symptoms, treatments, diagnoses)
- Generate structured medical summaries
- Analyze patient sentiment and intent
- Automatically generate SOAP notes

## Features

### 1. Named Entity Recognition (NER)
- Extracts symptoms, treatments, diagnoses, body parts, and temporal information
- Uses rule-based extraction with support for medical ML models
- Identifies key medical entities from unstructured text

### 2. Medical Summarization
- Converts raw transcripts into structured JSON format
- Extracts patient information, diagnosis, treatment, and prognosis
- Handles missing or ambiguous data gracefully

### 3. Sentiment & Intent Analysis
- Classifies patient sentiment as Anxious, Neutral, or Reassured
- Detects patient intent (seeking reassurance, reporting symptoms, etc.)
- Uses transformer-based approaches for healthcare-specific analysis

### 4. SOAP Note Generation (Bonus)
- Automatically generates structured SOAP notes
- Organizes information into Subjective, Objective, Assessment, and Plan sections
- Ensures clinical readability and completeness

## Requirements

### Base Requirements
```txt
Python >= 3.8
```

### For Full Production Deployment
```txt
spacy>=3.5.0
transformers>=4.30.0
torch>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
```

## Installation

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd physician-notetaker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install base requirements
pip install -r requirements.txt
```

### Full Installation (with ML models)

```bash
# Install full dependencies
pip install -r requirements-full.txt

# Download spaCy medical model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_med7_lg-0.5.1.tar.gz

# Or use standard spaCy model
python -m spacy download en_core_web_lg
```

## Usage

### Basic Usage

```python
from medical_nlp_pipeline import MedicalNLPPipeline

# Initialize pipeline
pipeline = MedicalNLPPipeline()

# Your medical transcript
transcript = """
Doctor: How are you feeling today?
Patient: I had a car accident. My neck and back hurt a lot for four weeks.
...
"""

# Extract entities
entities = pipeline.extract_entities(transcript)

# Generate summary
summary = pipeline.generate_summary(transcript, entities)

# Analyze sentiment
sentiment = pipeline.analyze_sentiment("I'm worried about my back pain")

# Generate SOAP note
soap_note = pipeline.generate_soap_note(transcript, entities)
```

### Command Line Usage

```bash
# Run the demonstration
python medical_nlp_pipeline.py

# Process a transcript file
python medical_nlp_pipeline.py --input transcript.txt --output results.json
```

## Expected Outputs

### 1. Entity Extraction
```json
{
  "symptoms": ["Neck pain", "Back pain", "Head impact"],
  "treatments": ["10 physiotherapy sessions", "Painkillers"],
  "diagnosis": ["Whiplash injury"],
  "body_parts": ["neck", "back", "head"],
  "temporal_info": ["four weeks", "six months"]
}
```

### 2. Medical Summary
```json
{
  "Patient_Name": "Janet Jones",
  "Symptoms": ["Neck pain", "Back pain", "Head impact"],
  "Diagnosis": "Whiplash injury",
  "Treatment": ["10 physiotherapy sessions", "Painkillers"],
  "Current_Status": "Occasional backache",
  "Prognosis": "Full recovery expected within six months"
}
```

### 3. Sentiment Analysis
```json
{
  "Sentiment": "Anxious",
  "Intent": "Seeking reassurance",
  "Confidence": 0.85
}
```

### 4. SOAP Note
```json
{
  "Subjective": {
    "Chief_Complaint": "Neck and back pain",
    "History_of_Present_Illness": "Patient had a car accident..."
  },
  "Objective": {
    "Physical_Exam": "Full range of motion...",
    "Observations": "Patient appears in normal health..."
  },
  "Assessment": {
    "Diagnosis": "Whiplash injury and lower back strain",
    "Severity": "Mild, improving"
  },
  "Plan": {
    "Treatment": "Continue physiotherapy...",
    "Follow-Up": "Patient to return if pain worsens..."
  }
}
```

## Technical Approach

### Handling Ambiguous/Missing Data

1. **Contextual Inference**: Use surrounding text to infer missing information
2. **Default Values**: Provide reasonable defaults for common medical fields
3. **Confidence Scores**: Include confidence levels for uncertain extractions
4. **Multi-pass Processing**: Run multiple extraction passes with different strategies
5. **Human-in-the-loop**: Flag uncertain extractions for manual review

### Pre-trained Models

#### For Medical NER:
- **ScispaCy's en_core_med7_lg**: Trained on medical entities
- **BioBERT**: BERT pre-trained on biomedical literature
- **ClinicalBERT**: BERT fine-tuned on clinical notes (MIMIC-III)
- **BlueBERT**: Pre-trained on PubMed and MIMIC-III

#### For Sentiment Analysis:
- **Bio_ClinicalBERT**: Medical sentiment classification
- **Mental-BERT**: Mental health-specific sentiment
- **Custom fine-tuned models**: On healthcare conversation datasets

#### For Summarization:
- **BART**: Fine-tuned on medical summaries
- **T5**: Adapted for medical text generation
- **LED (Longformer)**: For longer medical documents
- **BioGPT**: Medical text generation

### Fine-tuning BERT for Medical Sentiment

```python
# Example fine-tuning approach
from transformers import AutoModelForSequenceClassification, Trainer

# Load base medical BERT
model = AutoModelForSequenceClassification.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT",
    num_labels=3  # Anxious, Neutral, Reassured
)

# Prepare dataset
# - Medical Transcripts Dataset (if available)
# - MIMIC-III Notes with sentiment annotations
# - Custom annotated physician-patient conversations
# - Augmented data using GPT-based generation

# Fine-tune with medical-specific data
trainer = Trainer(
    model=model,
    train_dataset=medical_sentiment_train,
    eval_dataset=medical_sentiment_eval
)
trainer.train()
```

### Training Datasets

1. **MIMIC-III**: ICU clinical notes
2. **i2b2 Datasets**: Various medical NLP challenges
3. **PubMed Abstracts**: Medical literature
4. **Clinical Conversations**: Annotated physician-patient dialogues
5. **Custom Annotations**: Domain-specific labeled data

### SOAP Note Generation Approach

#### Rule-based Techniques:
- Keyword matching for section classification
- Template-based generation with slot filling
- Regular expressions for structured data extraction

#### Deep Learning Techniques:
- **Sequence-to-Sequence Models**: T5, BART for text-to-SOAP transformation
- **Extractive + Abstractive**: Combine extraction with generation
- **Multi-task Learning**: Train on multiple medical documentation tasks simultaneously
- **Few-shot Learning**: Use GPT-based models with examples

## Performance Optimization

- **Batch Processing**: Process multiple transcripts efficiently
- **Model Caching**: Cache loaded models to reduce initialization time
- **Parallel Processing**: Use multiprocessing for large datasets
- **GPU Acceleration**: Leverage CUDA for transformer models
- **Model Quantization**: Use lighter models for production deployment

## Future Enhancements

- [ ] Real-time transcription integration (speech-to-text)
- [ ] Multi-language support
- [ ] Integration with EHR systems
- [ ] Advanced medical reasoning (differential diagnosis)
- [ ] Clinical decision support
- [ ] Voice interface for hands-free operation
- [ ] Mobile application
- [ ] Cloud deployment (AWS/Azure/GCP)

## References

- Medical NER: [ScispaCy](https://allenai.github.io/scispacy/)
- Clinical BERT: [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
- SOAP Note Format: [Clinical Documentation Standards](https://www.aafp.org/pubs/fpm/issues/2017/0900/p27.html)
- MIMIC-III Dataset: [PhysioNet](https://physionet.org/content/mimiciii/)

## License

This project is for Emitrr case study assessment purposes only and is not intended for clinical use.

---
