# Causal Event Knowledge Graph (CEKG) Pipeline

> ⚠️ **EXPERIMENTAL BRANCH** ⚠️
>
> You are currently on the `experimental-features` branch. This branch contains unstable, in-development features and new experimental flags. For the stable, tested version, please check out the `main` branch.

This project is a Python-based preprocessing pipeline designed to read narrative text (e.g., "Great Expectations"), extract a Causal Event Knowledge Graph (CEKG) using an LLM, and export the graph into multiple formats for analysis.

## 🚀 Core Features

* **LLM-Powered Extraction:** Uses an LLM (e.g., `gpt-4o-mini`) to extract events, entities (actors, patients, locations, motivations), and causal links from plain text.
* **Modular & Extensible:** Refactored into a `cekg_pipeline` package where each module has a single responsibility (e.g., `llm_service`, `graph_builder`, `exporters`).
* **Multiple Exports:** Generates three types of output suitable for different graph databases and analysis tools:
    * JSON-LD
    * Neo4j Cypher script (.txt)
    * Neo4j Admin Import CSVs

---

## ✨ Experimental Features (This Branch Only)

This branch adds several powerful (and complex) experimental features, all controllable via command-line flags:

* **`--paragraph-chunk-size N`**: (Professor's "Idea-to-Idea") Groups `N` paragraphs into a single text chunk for one API call, forcing the LLM to find higher-level events instead of processing verb-by-verb.
* **`--graph-model "star"`**: (Professor's "Star Graph") Changes the output graph from the default `Event→Entity→Event` chain to a canonical `Entity → [Events]` star model.
* **`--enable-llm-expansion`**: (Group 2) Uses an advanced LLM prompt to perform coreference resolution (e.g., "the boy" → "Pip") and extract implicit/compound events.
* **`--enable-scene-grouping`**: (Group 1) Adds an extra LLM pass to group events into `Scene` nodes with a generated theme.
* **`--enable-semantic-linking`**: (Group 1) Adds an extra LLM pass to find non-causal `SemanticLink` relationships (e.g., "explanation", "elaboration").
* **`--enable-confidence-calibration`**: (Group 3) Replaces the LLM's simple `confidence` score with a calibrated formula (`pLLM + plexical + pcontextual`). **Requires `sentence-transformers`**.

---

## ⚙️ Setup & Installation

### 1. Prerequisites
* Python 3.10+
* Git

### 2. Install Dependencies
Clone the repository and install the required Python packages.

```bash
# Clone the repository (replace with your URL)
git clone [https://github.com/your-username/Causal-Event-Knowledge-Graph.git](https://github.com/your-username/Causal-Event-Knowledge-Graph.git)
cd Causal-Event-Knowledge-Graph

# IMPORTANT: Make sure you are on this branch
git checkout experimental-features

# Install dependencies (includes new ones for this branch)
pip install -r requirements.txt
````

**Note:** The `requirements.txt` for this branch must include:

```text
openai
pandas
python-dotenv
sentence-transformers
```

### 3\. Set Up API Key

Create a file named `.env` in the project's root directory:

**File: `.env`**

```text
# Add your secret API key
OPENAI_API_KEY="sk-YourSecretKeyGoesHere"

# You can optionally override the default model
OPENAI_MODEL="gpt-4o-mini"
```

-----

## 🏃 How to Run

All commands are run from the project's root directory using `main.py`.

### Standard (Default) Usage

Running the script without any experimental flags will use the **default "chain" model** and **parallel per-paragraph processing**, just like the `main` branch.

```bash
python main.py --input "Great Expectations.txt" --max-chapters 1
```

### Experimental Usage (All Features Enabled)

This command tests all new features at once. It will:

1.  Group 5 paragraphs at a time ("idea-to-idea").
2.  Output a "star" graph.
3.  Add `Scene` nodes and `SemanticLink` relationships.
4.  Use the expanded LLM prompt for coreference.
5.  Use the calibrated confidence score.

<!-- end list -->

```bash
python main.py \
  --input "Great Expectations.txt" \
  --max-chapters 1 \
\
  # --- Professor's "Idea-to-Idea" & "Star Graph" ---
  --paragraph-chunk-size 5 \
  --graph-model "star" \
\
  # --- Schema Features (Groups 1, 2, 3) ---
  --enable-scene-grouping \
  --enable-semantic-linking \
  --enable-llm-expansion \
  --enable-confidence-calibration
```

-----

## 📚 Command-Line Arguments

### Standard Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--input` | **(Required)** Path to the input `.txt` file. | N/A |
| `--out-json` | Path to save the JSON-LD output. | `ge_preprocessed.json` |
| `--out-cypher` | Base path for the Cypher `.txt` output. | `ge_import.cypher` |
| `--out-csv` | Directory to save the Neo4j CSV files. | `neo4j_csv` |
| `--max-chapters` | (Optional) Limit the number of chapters to process. | Processes all chapters |
| `--openai-model` | (Optional) The OpenAI model to use. | `gpt-4o-mini` |

### Experimental Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--paragraph-chunk-size` | Group N paragraphs for one "idea-to-idea" API call. | `1` (Per-paragraph) |
| `--graph-model` | Output graph model: `chain` (Event→Entity→Event) or `star` (Entity→[Events]). | `"chain"` |
| `--enable-scene-grouping` | Runs an extra LLM pass to create `Scene` nodes. | `False` |
| `--enable-semantic-linking` | Runs an extra LLM pass to find non-causal `SemanticLink` edges. | `False` |
| `--enable-llm-expansion` | Uses advanced prompt for coreference and implicit/compound events. | `False` |
| `--enable-confidence-calibration` | Calculates a 3-factor confidence score. **(Requires `sentence-transformers`)** | `False` |

```
```
