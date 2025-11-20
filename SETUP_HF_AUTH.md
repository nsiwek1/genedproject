# Hugging Face Authentication Setup

The Llama models are gated and require authentication. Follow these steps:

## 1. Get a Hugging Face Account & Token

1. Create an account at https://huggingface.co/join (if you don't have one)
2. Go to https://huggingface.co/settings/tokens
3. Create a new token with "Read" permissions
4. Copy the token (you'll need it in step 3)

## 2. Request Access to Llama Models

1. Visit https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
2. Click "Request access" or "Agree and access repository"
3. Wait for approval (usually instant or a few minutes)

## 3. Set Your Token

**Option A: Environment variable (recommended)**
```bash
export HF_TOKEN=your_token_here
```

To make it permanent, add to your `~/.zshrc`:
```bash
echo 'export HF_TOKEN=your_token_here' >> ~/.zshrc
source ~/.zshrc
```

**Option B: Pass as argument**
```bash
./run_training.sh --hf-token your_token_here
```

**Option C: Login interactively**
```bash
source .venv/bin/activate
huggingface-cli login
```
(Enter your token when prompted)

## 4. Verify Access

Try running the training script again:
```bash
./run_training.sh
```

If authentication works, you'll see "Authentication successful." and the model will download.

