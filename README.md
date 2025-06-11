# Document-chat-bot

This project implements a multi-turn conversational agent designed to answer queries based on an ESG (Environmental, Social, Governance) policy document. Built for non-technical users, it ensures responses are grounded in the source document. If a question is out of scope, the agent responds accordingly without generating unrelated content.

Features

Document-Aware Answering : Answers are generated using content from the uploaded ESG document via semantic search and an LLM.

Multi-turn Conversation : Maintains chat history to support follow-up questions and enhance context.

Guardrails : Filters unrelated queries using semantic similarity before calling the language model.

Answer Evaluation : Computes relevance, faithfulness, and confidence scores to indicate response quality.

User-Friendly Interface : Clean Streamlit-based UI for easy document upload and question-answer interaction.

Modular Design : Components are loosely coupled, allowing future changes to models or vector store backends.

Tech Stack LLM: Together.ai (lgai/exaone-deep-32b)

Embeddings: HuggingFace all-MiniLM-L6-v2

Vector Store: Chroma

Retriever: LangChain with page-wise document chunking

Memory: LangChain ConversationBufferMemory

Evaluation: Cosine similarity and LLM-based scoring

Evaluation Metrics

| **Metric**     | **Description**                                                                 |
|----------------|---------------------------------------------------------------------------------|
| Relevance      | Measures similarity between the user query and retrieved chunks                |
| Faithfulness   | Checks if the answer is grounded in the retrieved content                      |
| Confidence     | Combines relevance and faithfulness into a single reliability score            |

These metrics ensure responses remain accurate and aligned with the source document.


For each question, evaluation scores are tracked per session in a structured JSON format. This enables monitoring of chatbot performance over time and supports future improvements based on real usage data.

Setup Instructions & Requirements

Python 3.8 or above Together AI API Key

```python
# Install dependencies
pip install -r requirements.txt

# Configuration - Set your Together AI API key

# For Linux/Mac
export TOGETHER_API_KEY=your_api_key_here

# For Windows (Command Prompt)
set TOGETHER_API_KEY=your_api_key_here

# Run the App
streamlit run main.py

# The UI will launch in your browser.
# Upload your ESG document and start asking questions.
```

Limitations

Scalability: The current setup relies on local memory and storage, which may limit performance and flexibility in large-scale or multi-user deployments.

Inference Latency: Using free inference endpoints (e.g., lgai/exaone-deep-32b) and Hugging Face embedding models may introduce latency due to limited compute resources.

Scope for Improvement

Vector-Based Memory: Transition from in-memory buffers to vector-based memory or external memory stores with vector indexing. This enables persistent and scalable multi-turn conversation handling.

Cloud-Based Model Inference: Use cloud-hosted language models such as OpenAI (via Azure OpenAI Service) or Anthropic Claude (via AWS Bedrock) to ensure high availability, scalability and better performance

Managed Vector Database Support: Replace Chroma with enterprise-grade vector databases like Azure Cognitive Search for improved query performance, observability, and integration with cloud-native services.
