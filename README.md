# Agentic-rag-deepseek_

## Overview
This project implements an **Agentic Retrieval-Augmented Generation (RAG) System** using **LangChain** and **FAISS** for document retrieval and reasoning. The system dynamically decides when to retrieve context from documents and when to generate answers using an **LLM-powered agent**.

## Features
- Converts **PDF documents** into **Markdown format** for processing.
- Splits text into **structured chunks** for efficient retrieval.
- Uses **FAISS vector store** for semantic search.
- Implements **LangChain Agents** to dynamically decide when to retrieve or generate responses.
- Includes **math computation tools** (`llm-math`) alongside document retrieval.
- Uses **DeepSeek R1 model** for reasoning and text generation.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com//NithiN-1808/agentic-rag-deepseek_.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up **Ollama** locally:
   - Download and install [Ollama](https://ollama.com/)
   - Start the server:
     ```bash
     ollama serve
     ```
4. Create a `.env` file and set required environment variables:
   ```env
   OPENAI_API_KEY=your-api-key
   ```

## How It Works

### 1. Document Processing
- The system **loads and converts** a PDF document into **Markdown format** using `DocumentConverter`.
- It **splits** the Markdown content into structured chunks using `MarkdownHeaderTextSplitter`.
- It **embeds** these chunks using `OllamaEmbeddings` and stores them in **FAISS vector store**.

### 2. Agentic Reasoning and Retrieval
- A **LangChain Agent** (`ZERO_SHOT_REACT_DESCRIPTION`) is created.
- The agent has access to two tools:
  1. **Document Retrieval Tool** - Retrieves relevant document chunks.
  2. **Math Tool (`llm-math`)** - Solves mathematical queries.
- The agent **decides** dynamically whether to retrieve information from documents or generate an answer using reasoning.

### 3. Query Execution
- Users can ask questions, and the **agent** will:
  1. Decide **if retrieval is necessary**.
  2. Retrieve relevant document chunks if needed.
  3. Generate a **structured response** using an LLM.

## Usage

To run the system:
```bash
python finance_rag.py
```

### Example Questions
- "How much revenue is there for Google?"
- "What is the net income for this quarter, and what are the key drivers?"
- "Which business segment contributed the most to revenue?"

## Technologies Used
- **LangChain** (Agents, Tools, Embeddings, Retrieval)
- **FAISS** (Vector Store for Semantic Search)
- **DeepSeek R1 (1.5B)** (LLM for text generation)
- **Ollama** (Embeddings & Local LLM Serving)
- **Python** (Core logic and integrations)

## Future Improvements
- Add support for **multi-document retrieval**.
- Improve **agent prompting** for better decision-making.
- Integrate **more advanced embedding models** for better retrieval accuracy.

## License
This project is licensed under the MIT License.

