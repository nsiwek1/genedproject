## Overview

This workspace prepares textual datasets from the Analects and the Mencius so you can fine‑tune lightweight Llama models that speak in the voices of Confucius and Mencius. The pipeline covers:

- segmenting each classical text into labeled, citation-preserving snippets
- authoring instruction–response chat pairs per philosopher
- outlining LoRA fine-tuning with Hugging Face PEFT/Transformers
- wiring an interactive Gradio UI that routes questions to the appropriate persona

The artistic goal is to let users pose identical prompts to contrasting philosophers and surface their divergent notions of ritual, human nature, and governance.

## 1. Data Preparation

### Segment the source texts

```
python3 /Users/natalia_mac/gened_p/scripts/segment_texts.py
```

Outputs:

- `data/segments/confucius_segments.jsonl`
- `data/segments/mencius_segments.jsonl`
- `data/segments/all_segments.jsonl`

Each JSON line contains:

```
{
  "philosopher": "Confucius",
  "work": "Analects",
  "book": "Two",
  "reference": "2.3",
  "text": "..."
}
```

The script removes page headers, footnotes, and PDF hyphenation, giving you clean, citation-tagged snippets that can be sampled into downstream datasets. If you add new texts, extend `SourceConfig` inside `scripts/segment_texts.py`.

### Optional interpretive paraphrases

To enrich each snippet with a modern-language summary, add a new field (e.g., `interpretation`) to the JSON objects. You can write these manually, or script a pass that consumes each `text` and appends a paraphrase column (e.g., via a spreadsheet or a lightweight semantic model). Keep the literal translation separate so you can trace every generated sentence back to a primary passage.

## 2. Instruction–Response Dataset

Sample persona-aligned conversations live in:

- `data/instruction_pairs/confucius_sample.jsonl`
- `data/instruction_pairs/mencius_sample.jsonl`

Each record follows the OpenAI-style `messages` schema and stores a `source_reference` for attribution. Use these as templates when expanding to 50–200 dialogues per philosopher:

1. Pick a segment from `data/segments/...`.
2. Draft a user question that naturally elicits that teaching.
3. Write the assistant reply in the philosopher’s cadence, weaving in the cited doctrine.
4. Record the `source_reference` to keep responses auditable.

Tips:

- Maintain distinct `system` prompts per philosopher so the model learns different registers (ritualist vs. innate goodness).
- Mix prompt types: leadership dilemmas, self-cultivation, metaphors, edge cases.
- Interleave historical context questions with practical advice to avoid overly narrow conditioning.

When the dataset scales up, merge both philosophers into a single JSONL for training, optionally prepending a tag like `[Confucius]`/`[Mencius]` to the user turn if you prefer one multi-persona checkpoint.

## 3. Fine-Tuning Plan (LoRA + Transformers)

1. **Environment**
   ```
   pip install --upgrade torch transformers peft accelerate datasets
   ```
2. **Dataset ingestion**
   ```python
   from datasets import load_dataset
   data = load_dataset("json", data_files={
       "train": "/Users/natalia_mac/gened_p/data/instruction_pairs/confucius_sample.jsonl"
   })
   ```
3. **Model choice**
   - `meta-llama/Llama-3.2-1B-Instruct` for fast iteration
   - `meta-llama/Llama-3.2-3B-Instruct` or `8B` when you need richer prose (requires ≥24 GB VRAM for full fine-tune; LoRA keeps it lightweight)
4. **LoRA recipe (one checkpoint per philosopher)**
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from peft import LoraConfig, get_peft_model

   model = AutoModelForCausalLM.from_pretrained(base_model, load_in_8bit=True, device_map="auto")
   tokenizer = AutoTokenizer.from_pretrained(base_model)
   lora_cfg = LoraConfig(
       r=64, lora_alpha=128, lora_dropout=0.1,
       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
   )
   model = get_peft_model(model, lora_cfg)
   ```
5. **Training loop**
   - Use `transformers.Trainer` or `trl.SFTTrainer`
   - Sequence length: 1,024 tokens (the sayings are short)
   - Batch: 64 tokens/step effective (gradient accumulation if needed)
   - Learning rate: 2e-4 with cosine decay, 2–3 epochs
   - Save adapter weights under `models/confucius-lora/`, `models/mencius-lora/`
6. **Single blended model option**
   - Merge both datasets
   - Add tags like `[Confucius]` to user inputs during training
   - At inference, prepend the chosen tag to the prompt

## 4. Interactive Chat Interface

Prototype (switches philosophers via dropdown):

```python
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizers = {
    "Confucius": AutoTokenizer.from_pretrained("models/confucius-lora"),
    "Mencius": AutoTokenizer.from_pretrained("models/mencius-lora"),
}
models = {
    name: AutoModelForCausalLM.from_pretrained(path).eval()
    for name, path in [
        ("Confucius", "models/confucius-lora"),
        ("Mencius", "models/mencius-lora"),
    ]
}

def answer(philosopher, question):
    tokenizer = tokenizers[philosopher]
    model = models[philosopher]
    prompt = f"[{philosopher}] {question}"
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    return tokenizer.decode(output[0], skip_special_tokens=True)

gr.Interface(
    fn=answer,
    inputs=[gr.Dropdown(["Confucius", "Mencius"]), gr.Textbox(label="Ask a question")],
    outputs="text",
    title="Chinese Philosophy Chat",
    description="Contrast Confucian ritualism with Mencian moral sprouts."
).launch()
```

If you train a single blended model, keep one tokenizer/model pair and simply prepend the dropdown value inside `answer`.

## 5. Next Steps & Evaluation

- Scale instruction pairs to ≥50 per philosopher, mixing question types.
- Add interpretive paraphrases to `segments` so the model learns both literal citations and modern commentary.
- Track provenance: keep `source_reference` so generated answers can display citations.
- Design evaluation prompts (e.g., moral dilemmas) and compare generated replies to reference summaries; manually score for doctrinal accuracy and tone.

With this pipeline, you can now iterate from raw texts to persona-specific adapters and an interactive experience that reveals the contrasts among early Chinese thinkers.

