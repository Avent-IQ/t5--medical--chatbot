# Text-to-Text Transfer Transformer (T5) Quantized Model for Medical Chatbot

This repository hosts a quantized version of the T5 model, fine-tuned for Medical Chatbot  tasks. The model has been optimized for efficient deployment while maintaining high accuracy, making it suitable for resource-constrained environments.

## Model Details
- **Model Architecture:** T5  
- **Task:** Medical Chatbot  
- **Dataset:** Hugging Face's ‘medical-qa-datasets’
 
- **Quantization:** Float16
- **Fine-tuning Framework:** Hugging Face Transformers  

## Usage
### Installation
```sh
pip install transformers torch
```

### Loading the Model
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "AventIQ-AI/t5-medical-chatbot”
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

def test_medical_t5(instruction, input_text, model, tokenizer):
    """Format input like the training dataset and test the quantized model."""
    formatted_input = f"Instruction: {instruction} Input: {input_text}"
 
    # ✅ Tokenize input & move to correct device
    inputs = tokenizer(
        formatted_input, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)
 
    # ✅ Generate response with optimized settings
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],  # Explicitly specify input tensor
            attention_mask=inputs["attention_mask"],
            max_length=200,
            num_return_sequences=1,
            temperature=0.6,
            top_k=40,
            top_p=0.85,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
 
    # ✅ Decode output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Test Example
instruction = "As a medical expert, provide a detailed and accurate diagnosis based on the patient's symptoms."
input_text = "A patient is experiencing persistent hair fall, dizziness, and nausea. What could be the underlying cause and recommended next steps?"
```

## 📊 ROUGE Evaluation Results
After fine-tuning the T5-Small model for Medical Chatbot, we obtained the following ROUGE scores:

| **Metric**  | **Score** | **Meaning**  |
|------------|---------|--------------------------------------------------------------|
| **ROUGE-1**  | 1.0 (~100%) | Measures overlap of unigrams (single words) between the reference and generated text. |
| **ROUGE-2**  | 0.5 (~50%) | Measures overlap of bigrams (two-word phrases), indicating coherence and fluency. |
| **ROUGE-L**  | 1.0 (~100%) | Measures longest matching word sequences, testing sentence structure preservation. |
| **ROUGE-Lsum**  | 0.95 (~95%) | Similar to ROUGE-L but optimized for summarization tasks. |

## Fine-Tuning Details
### Dataset
The Hugging Face's `medical-qa-datasets’ dataset was used, containing different types of Patient and Doctor Questions and respective Answers.

### Training
- **Number of epochs:** 3  
- **Batch size:** 8  
- **Evaluation strategy:** epoch  

### Quantization
Post-training quantization was applied using PyTorch's built-in quantization framework to reduce the model size and improve inference efficiency.

## Repository Structure
```
.
├── model/               # Contains the quantized model files
├── tokenizer_config/    # Tokenizer configuration and vocabulary files
├── model.safetensors/   # Quantized Model
├── README.md            # Model documentation
```

## Limitations
- The model may not generalize well to domains outside the fine-tuning dataset.
- Currently, it only supports English to French translations.
- Quantization may result in minor accuracy degradation compared to full-precision models.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.
