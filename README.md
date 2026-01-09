# Autonym 

**Automated ML Run Naming & Summarization using LLMs.**

This is a vibe-coded tool to help me track my protoyping experiments. Autonym experiment scribe reads your current `git diff`, separates logic changes from config updates, and uses an LLM (Local or Remote) to generate a semantic `run_name` and technical `description`.

It solves the problem of manual run tracking ("run-42", "run-final-final") by ensuring every experiment has a descriptive, auto-generated label based on what changed in the code.

![WandB Screenshot](wandb_screenshot.png)

## Features

- **Smart Diffing:** Separates code logic from YAML config changes.
- **Grammar Forcing:** Uses Pydantic grammar forcing to output valid JSON to avoid excessively long responses.
- **Provider Agnostic:** Supports local **Ollama** (free, private) and **OpenAI** (remote, fast).

## Usage
```python
import wandb
from autonym import Autonym

# ... config setup ...
# ... argparse, yaml load ...

scribe = Autonym(provider="ollama", model_name="phi3.5")

# Generate run metadata from git diffs & config changes
meta = scribe.summarize_run(runtime_config=config, config_path="config.yaml", base_ref="origin/main")

if meta:
    wandb.init(name=meta.run_name, notes=meta.description, config=config)
else:
    wandb.init(config=config)

# ... training loop ...
train(config)
```


## Installation

1. Install Autonym
```bash
# 1. Install Autonym
pip install autonym
```

2. For local inference, install [ollama](https://ollama.com/download). Phi3.5 was good enough for me on an M1 Mac.
```bash
ollama pull phi3.5
```