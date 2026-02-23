# ISO/IEC 27001 Compliance RAG Assistant
A compliance-grade Retrieval-Augmented Generation (RAG) system for answering ISO/IEC 27001:2022 questions without hallucinations, backed by automated evaluation and traceability.

## What This Project Does
This system allows users to ask ISO/IEC 27001:2022 compliance questions and receive answers that:
- are strictly grounded in the standard
- use the standard’s normative language (“shall”, not “should”)
- refuse to answer when the document does not contain the information
- provide traceable retrieval evidence
- are automatically evaluated for quality using RAGAS
It is designed for audit, compliance, and security use cases, where hallucinations are unacceptable.

## Why This Matters
In compliance and audit workflows:
- A confident but incorrect answer is worse than no answer.
- Answers must be verifiable, traceable, and auditable.
- Most AI demos ignore evaluation, this project treats it as first-class.

This project demonstrates how to build trustworthy AI for regulated domains.

## System Architecture
User Question -> Vector Retrieval (Pinecone) -> Reranking (Cohere) -> Grounded Answer Generation (OpenAI) -> Traceability (LangSmith) -> Automated Evaluation (RAGAS)

## Evaluation & Quality Assurance

The system is evaluated using RAGAS, focusing on four key metrics:

- Faithfulness – Is the answer fully supported by retrieved content?
- Answer Relevancy – Does the answer directly address the question?
- Context Precision – Were retrieved documents relevant?
- Context Recall – Was all required information retrieved?

```
faithfulness:        1.0000
answer_relevancy:    0.9037
context_precision:   1.0000
context_recall:      1.0000
```

## Tech Stack
- Python
- LangChain
- OpenAI (GPT-4o-mini, embeddings)
- Pinecone (vector storage)
- Cohere Rerank
- RAGAS (evaluation)
- LangSmith (tracing & observability)
- Streamlit (UI)

## Future Improvements
- Expand evaluation dataset (20–50 questions)
- Add multi-document support (ISO 27002, ISO 27005)
- Store evaluation results in a database for trend analysis
- Add citation highlighting in UI
- Support audit-style evidence export

## How to Run Locally

This project is designed to run locally using Python and environment variables for API access.
You can either use uv or pip.

### 1. Clone the repository
- git clone https://github.com/GraciousWeb/professional_RAG_assistant.git
- cd professional_RAG_assistant

### 2. Create and activate a virtual environment (recommended)
```
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

All required dependencies are listed in requirements.txt.

`pip install -r requirements.txt`

### 4. Set up environment variables
Create a .env file in the project root and add the following:

```OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name
COHERE_API_KEY=your_cohere_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=iso27001-rag
```

### 5. Run the Streamlit demo

Launch the interactive UI:

`streamlit run streamlit_app.py`

### 6. Run automated evaluation (RAGAS)

To evaluate answer quality using RAGAS:

`python evaluate_auditor.py`

This will:

- run the RAG pipeline against a predefined evaluation dataset

- compute faithfulness, relevancy, and retrieval metrics

- store results locally under ragas_data/

- print aggregated scores to the console

Example output:

```
faithfulness: 1.0000
answer_relevancy: 0.9037
context_precision: 1.0000
context_recall: 1.0000
```

### Optional: Observability with LangSmith

If LANGCHAIN_TRACING_V2=true and a valid LANGSMITH_API_KEY are set, all runs will be traced automatically.

LangSmith traces allow you to inspect:

- retrieval results

- reranking decisions

- prompt construction

- model responses

This is strongly recommended when experimenting with retrieval parameters or prompt changes.