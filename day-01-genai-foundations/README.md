# Day 1 Assignment - Text Generation Tool

A CLI-based text generation tool that compares open-source (HuggingFace) and API-based LLM models at different temperature settings.

## Features

- ✅ Accepts user input via command line or interactive prompt
- ✅ Generates text using HuggingFace open-source models
- ✅ Generates text using API-based models (OpenAI/Groq)
- ✅ Tests 3 different temperature values (0.2, 0.7, 1.2)
- ✅ Saves structured outputs to JSON
- ✅ Compares model behavior across temperatures

## Project Structure

```
day-01-genai-foundations/
├── src/
│   ├── text_generation.py  # Entry point
│   ├── hf_model.py         # HuggingFace model logic
│   ├── api_model.py        # API-based model logic
│   └── config.py           # Configuration settings
├── outputs/
│   └── sample_outputs.json # Generated outputs
├── requirements.txt        # Python dependencies
├── .env                    # API keys (create from .env.example)
└── README.md              # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API key:
   ```
   OPENAI_API_KEY=your_actual_key_here
   ```
   
   Or for Groq:
   ```
   GROQ_API_KEY=your_actual_key_here
   ```

3. Update `src/config.py` to set your preferred API provider:
   ```python
   API_PROVIDER = "openai"  # or "groq"
   ```

### 3. Run the Tool

**Option 1: Interactive mode**
```bash
python src/text_generation.py
```

**Option 2: Command-line argument**
```bash
python src/text_generation.py "Your prompt here"
```

## Output Format

The tool generates a JSON file at `outputs/sample_outputs.json` with the following structure:

```json
{
  "input": "Your prompt here",
  "timestamp": "2024-01-01T12:00:00",
  "outputs": {
    "huggingface": {
      "temp=0.2": "Generated text...",
      "temp=0.7": "Generated text...",
      "temp=1.2": "Generated text..."
    },
    "api_model": {
      "temp=0.2": "Generated text...",
      "temp=0.7": "Generated text...",
      "temp=1.2": "Generated text..."
    }
  }
}
```

## Configuration

Edit `src/config.py` to customize:

- **Temperature values**: `TEMPERATURES = [0.2, 0.7, 1.2]`
- **HuggingFace model**: `HF_MODEL_NAME = "gpt2"`
- **API provider**: `API_PROVIDER = "openai"`
- **API model**: `API_MODEL_NAME = "gpt-3.5-turbo"`
- **Max tokens**: `MAX_NEW_TOKENS = 150`

## Notes

- The HuggingFace model will download on first run (may take a few minutes)
- GPU is automatically used if available (CUDA)
- Make sure you have sufficient API credits for the API provider you choose
- Lower temperatures (0.2) produce more deterministic outputs
- Higher temperatures (1.2) produce more creative/varied outputs

## Troubleshooting

**Error: API key not found**
- Make sure you created `.env` file from `.env.example`
- Verify your API key is correct

**Error: Model loading failed**
- Check your internet connection (models download from HuggingFace)
- Try a smaller model like "distilgpt2" in `config.py`

**Error: CUDA out of memory**
- The code automatically falls back to CPU if GPU fails
- Or use a smaller model in `config.py`

