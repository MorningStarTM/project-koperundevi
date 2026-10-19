# Tokenizer vocab-size sweep

Trains 3 BPE tokenizers (vocab sizes 6000 / 8000 / 10000) on the same 50MB
sample of `data_tamil.txt`, each as a separate Pulumi stack / EC2 instance,
each pushed to `ErnestBeckham/gpt-v1` on Hugging Face Hub as
`tokenizer_<vocab_size>.json`. Same data sample + seed across all three, so
vocab_size is the only variable -- a fair comparison. See `__main__.py`'s
docstring for the full design notes.

## Before running

Check your current public IP (it tends to drift) and use it for `sshCidr`
in every stack below:

```powershell
(Invoke-WebRequest -Uri "https://api.ipify.org" -UseBasicParsing).Content
```

`.env` at the repo root must have `KAGGLE_USERNAME`, `KAGGLE_KEY`, `HF_TOKEN`
set -- these are read automatically, no need to pass them as config.

## One-time setup

```powershell
cd infra_tokenizer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Deploy all three stacks

```powershell
pulumi stack init vocab6k
pulumi config set aws:region eu-north-1
pulumi config set keyName gridstar-key
pulumi config set sshCidr "<your current IP>/32"
pulumi config set kaggleDataset ernestbeckham/tamil-corpus
pulumi config set vocabSize 6000
pulumi up

pulumi stack init vocab8k
pulumi config set aws:region eu-north-1
pulumi config set keyName gridstar-key
pulumi config set sshCidr "<your current IP>/32"
pulumi config set kaggleDataset ernestbeckham/tamil-corpus
pulumi config set vocabSize 8000
pulumi up

pulumi stack init vocab10k
pulumi config set aws:region eu-north-1
pulumi config set keyName gridstar-key
pulumi config set sshCidr "<your current IP>/32"
pulumi config set kaggleDataset ernestbeckham/tamil-corpus
pulumi config set vocabSize 10000
pulumi up
```

Confirm `yes` on each `pulumi up`.

## Checking progress

Each stack creates its own instance. Switch stacks with
`pulumi stack select <name>` before checking that stack's instance.

Via Session Manager (no SSH, no IP allowlist needed, works immediately):

```powershell
pulumi stack output instance_id
aws ssm start-session --target <instance_id> --region eu-north-1
```

Inside the session:

```bash
sudo su -
cat /root/DONE 2>/dev/null && echo DONE || echo "still running"
tail -f /var/log/bpe-bootstrap.log
```

Or via plain SSH (re-sync `sshCidr` first if your IP has drifted):

```powershell
pulumi stack output ssh
```

(prints a ready-to-run `ssh -i ...` command)

## After training: evaluate

Use `gpt/eval_tokenizer.py` with `--hf-repo-id` and `--hf-filename` to pull
each tokenizer straight from HF Hub, e.g.:

```python
!python gpt/eval_tokenizer.py \
    --hf-repo-id ErnestBeckham/gpt-v1 \
    --hf-filename tokenizer_6000.json \
    --hf-token <your HF token> \
    --eval-data <held-out data_tamil.txt slice> \
    --max-chars 2000000
```

Repeat for `tokenizer_8000.json` and `tokenizer_10000.json`, compare
compression ratio / fertility / vocab utilization, then pick one.

## Cleanup (stop billing)

```powershell
pulumi stack select vocab6k
pulumi destroy

pulumi stack select vocab8k
pulumi destroy

pulumi stack select vocab10k
pulumi destroy
```
