# Causal Event Knowledge Graph (CEKG) Pipeline

This project is a Python-based preprocessing pipeline designed to read narrative text (e.g., "Great Expectations"), extract a Causal Event Knowledge Graph (CEKG) using an LLM, and export the graph into multiple formats for analysis.

## 🚀 Key Features

  * **LLM-Powered Extraction:** Uses an LLM (e.g., `gpt-4o-mini`) to extract events, entities (actors, patients, locations, motivations), and causal links from plain text.
  * **Dual Flow Architecture:** Builds a directed graph based on a "Dual Flow" model (`Event₁ → Entity → Event₂`) to create a clear, directional chain of actions.
  * **Modular & Extensible:** Refactored into a `cekg_pipeline` package where each module has a single responsibility (e.g., `llm_service`, `graph_builder`, `exporters`).
  * **Multiple Exports:** Generates three types of output suitable for different graph databases and analysis tools:
      * JSON-LD
      * Neo4j Cypher script (.txt)
      * Neo4j Admin Import CSVs

## 🏛️ Core Architecture: Dual Flow

To maintain a directed, acyclic graph (DAG) structure, this pipeline avoids direct `Entity-[:PARTICIPATES_IN]->Event` relationships. Instead, it uses a "Dual Flow" model:

1.  **Event Produces Entity:** An event is the source of all entities involved in it (e.g., `Event -[:PRODUCES_ACTOR]-> Agent`).
2.  **Entity Points to Next Event:** An entity instance then points to the *next* event it participates in (e.g., `Agent -[:ACTS_IN]-> NextEvent`).

This creates a clear, directional chain: **`Event₁ → Entity → Event₂ → Entity → Event₃`**

## 📂 Project Structure

```
Causal-Event-Knowledge-Graph/
├── cekg_pipeline/
│   ├── __init__.py           # Makes this a Python package
│   ├── config.py             # Constants and settings (model names, etc.)
│   ├── schemas.py            # All dataclasses (CEKEvent, GenericNode, etc.)
│   ├── utils.py              # Cache, DAGValidator, and other helpers
│   ├── text_processor.py     # Text loading and chapter/paragraph splitting
│   ├── llm_service.py        # All OpenAI API calls and JSON cleaning
│   ├── graph_builder.py      # Core logic (context propagation, linking)
│   ├── graph_mapper.py       # Maps specific data to a generic graph
│   └── exporters.py          # Exports to Cypher, JSON, and CSV
│
├── main.py                   # Main script to run the pipeline
├── requirements.txt          # Python dependencies
├── .env                      # (Your local file) For API keys
└── Great Expectations.txt      # Example input data
```

## ⚙️ Setup & Installation

### 1\. Prerequisites

  * Python 3.10+
  * Git

### 2\. Install Dependencies

Clone the repository and install the required Python packages:

```bash
# Clone the repository (replace with your URL)
git clone https://github.com/your-username/Causal-Event-Knowledge-Graph.git
cd Causal-Event-Knowledge-Graph

# Install dependencies
pip install -r requirements.txt
```

### 3\. Set Up API Key

This project requires an OpenAI API key. Create a file named `.env` in the project's root directory:

**File: `.env`**

```text
# Add your secret API key
OPENAI_API_KEY="sk-YourSecretKeyGoesHere"

# You can optionally override the default model
OPENAI_MODEL="gpt-4o-mini"
```

## 🏃 How to Run

All commands are run from the project's root directory using `main.py`.

### Basic Example

This command processes the first 5 chapters of "Great Expectations" and saves the default output files.

```bash
python main.py --input "Great Expectations.txt" --max-chapters 5
```

### Full Command Example

This command specifies all output paths and uses a different model.

```bash
python main.py \
    --input "Great Expectations.txt" \
    --out-json "my_graph.json" \
    --out-cypher "my_graph_import" \
    --out-csv "my_csv_files/" \
    --max-chapters 10 \
    --openai-model "gpt-4o"
```

### Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--input` | **(Required)** Path to the input `.txt` file. | N/A |
| `--out-json` | Path to save the JSON-LD output. | `ge_preprocessed.json` |
| `--out-cypher` | Base path for the Cypher `.txt` output. | `ge_import.cypher` |
| `--out-csv` | Directory to save the Neo4j CSV files. | `neo4j_csv` |
| `--max-chapters` | (Optional) Limit the number of chapters to process. | Processes all chapters |
| `--openai-model` | (Optional) The OpenAI model to use. | `gpt-4o-mini` |
| `--batch-size` | (Optional) Paragraphs to batch for event extraction. | 5 |
| `--causal-batch-size` | (Optional) Causal pairs to batch for assessment. | 10 |

## 📊 Outputs

After running the script, you will find the following outputs in your project directory:

1.  **`ge_preprocessed.json`** (or your `--out-json` name)

      * A single JSON-LD file representing the full graph, including all nodes and relationships.

2.  **`ge_import.txt`** (or your `--out-cypher` name)

      * A complete Neo4j Cypher script. You can paste this directly into the Neo4j browser to create the entire graph. It uses `MERGE` to be idempotent.

3.  **`neo4j_csv/`** (or your `--out-csv` name)

      * A directory containing multiple CSV files (e.g., `events.csv`, `agents.csv`, `produces_actor.csv`). This format is optimized for high-speed bulk import using the `neo4j-admin database import` command.
