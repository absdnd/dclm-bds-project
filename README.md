# 📦 Dataset Deduplication Pipeline

## 🛠️ Setup

### 2. Install Dependencies

This project uses [Poetry](https://python-poetry.org/) for dependency management:

```bash
poetry install
```

### 3. Set up Environment Variables

Copy the example `.env` file and edit the variables:

```bash
cp .env.exact.example .env
```

**Example `.env` file:**

```env
DATASET_NAME=c4
DATASET_CONFIG=en.noclean
DATASET_SPLIT=train
TEXT_COLUMN=text
METHOD=exact
CHUNK_SIZE=10000 ## chunk size to stream from hf
MAX_CHUNKS=10

WANDB_PROJECT=dedup-benchmark

## repo where the deduped dataset is pushed
HF_REPO_ID=your_repo_name
HF_PRIVATE=false
HF_TOKEN=your_token
```

### 4. Activate the Environment

```bash
poetry shell
```

---

## 🚀 Run the Pipeline

Once the `.env` is set up:

```bash
python scripts/run_pipeline.py
```

---

You can switch techniques by changing `METHOD` in your `.env` file.

---

## 📈 Logging with Weights & Biases

- Logs runtime, memory usage, and duplicates removed.
- Optionally visualizes metrics like compression ratio and similarity histograms.

To enable logging, set your W&B project name in `.env`:

```env
WANDB_PROJECT=dedup-benchmark
```

Then log in:

```bash
wandb login
```

---

## 🧱 Project Structure

```
dedup/
├── base.py             # Base class for all deduplicators
├── bloomfilter.py      # Bloom Filter Deduplicator
├── exact.py        # ExactHash deduplicator
├── minhash.py          # MinHash LSH deduplicator
├── metrics/     # Metrics collection
├── utils/
├── config.py           # Pydantic settings from .env
├── runner.py
scripts/
├── run_pipeline.py        # Main pipeline script
.env                    # Configuration file
```

---

## ➕ Add Your Own Deduplicator

1. Create a new file in `dedup/`, e.g. `yourtechnique.py`.
2. Implement a class that inherits from `Deduplicator`:

```python
from dedup.base import Deduplicator

class YourDeduplicator(Deduplicator):
    def run(self, examples: list[dict]) -> list[dict]:
        # your logic here
        return deduped_examples
```

3. Register it in `dedup/registry.py`:

```python
from .yourtechnique import YourDeduplicator

DEDUPLICATOR_REGISTRY = {
    "exact": ExactHashDeduplicator,
    "bloom": BloomFilterDeduplicator,
    "minhash": MinHashDeduplicator,
    "yourtechnique": YourDeduplicator,
}
```

4. Use it by setting `METHOD=yourtechnique` in your `.env`.

---

## 📤 Save to Hugging Face Hub (Optional)

Set these in `.env`:

```env
HF_REPO_ID=your_repo_name
HF_PRIVATE=true
HF_TOKEN=your_token
```

This enables pushing the deduplicated dataset to the Hub after processing.

---

## 🧪 Tested On

- ✅ `c4/en.noclean`

Supports streaming + chunked deduplication for large datasets.

---

## 🧭 Future Roadmap

- [x] `.env`-based config loading
- [x] Exact, Bloom, and MinHash deduplicators
- [x] W&B metrics + visualizations
- [ ] Parallelized deduplication with multiprocessing/Spark
- [ ] Hugging Face Hub upload via `datasets.Dataset.push_to_hub`
- [ ] Deduplicator ensemble analysis (Venn overlap, shared hits)

---
