import os
from langchain.vectorstores import Chroma
from langchain_together import Together
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np

os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")

def is_query_related_to_esg(
    query: str, retriever, embedding, threshold: float = 0.4
) -> bool:
    """
    Checks if a user query is related to the ESG document using embedding similarity.

    Args:
        query (str): User question.
        retriever: Vector store retriever.
        embedding: Embedding model.
        threshold (float): Minimum similarity threshold to consider the query relevant.

    Returns:
        bool: True if the query is related, False otherwise.
    """
    query_emb = np.array(embedding.embed_query(query)).reshape(1, -1)
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return False

    doc_embs = np.array([embedding.embed_query(d.page_content) for d in docs])
    sims = cosine_similarity(query_emb, doc_embs)[0]
    max_sim = np.max(sims)

    return max_sim >= threshold


def configure_model():
    """
    Configures the Together LLM for answering queries.

    Returns:
        Together: Configured language model instance.
    """
    return Together(
        model="lgai/exaone-deep-32b",
        temperature=0.1,
        max_tokens=3000,
        top_p=0.7,
    )


def rough_confidence_score(
    relevance, faithfulness, weight_relevance=0.5, weight_faithfulness=0.5
):
    """
    Combines relevance and faithfulness into a single confidence score"""
    return weight_relevance * relevance + weight_faithfulness * faithfulness


def evaluate_answer(llm, query, answer, docs, embedding):
    """
    Evaluates an answer based on relevance to the query and faithfulness to retrieved documents.

    Args:
        llm: Language model used for evaluation.
        query (str): User query.
        answer (str): Model-generated answer.
        docs (List[str]): Retrieved documents.
        embedding: Embedding model.

    Returns:
        dict: Dictionary with relevance, faithfulness, and confidence scores.
    """
    query_emb = np.array(embedding.embed_query(query)).reshape(1, -1)
    doc_embs = np.array([embedding.embed_query(doc) for doc in docs])
    relevance = (
        float(np.max(cosine_similarity(query_emb, doc_embs)[0]))
        if len(doc_embs)
        else 0.0
    )
    prompt = f"""
    You are an expert evaluator. Determine whether the given answer is faithful to the retrieved documents.

    Question: {query}

    Answer:
    {answer}

    Retrieved Documents:
    {"".join(docs)}

    Faithfulness Criteria:
    - The answer should be directly supported by the documents.
    - No hallucinated or invented facts.
    - Partial matches should score less than 1.0

    Rate the faithfulness on a scale of 0 to 1.
    Return ONLY a single number.
    """
    resp = llm.invoke(prompt)
    match = re.findall(r"\d\.\d+", resp)
    faithfulness = float(match[0]) if match else 0.0
    confidence = rough_confidence_score(relevance, faithfulness)
    return {
        "relevance": relevance,
        "faithfulness": faithfulness,
        "confidence": confidence,
    }


def run_agent(query: str):
    """
    Runs the ESG QA agent on a user query with document retrieval and evaluation.

    Args:
        query (str): User question.

    Returns:
        Tuple[str, dict]: Answer and a dictionary of evaluation scores.
    """
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(persist_directory="vector_store", embedding_function=embedding)
    retriever = vectorstore.as_retriever()
    if not is_query_related_to_esg(query, retriever, embedding):
        return "This question is not related to the ESG policy document.", {
            "relevance": 0.0,
            "faithfulness": 0.0,
            "confidence": 0.0,
        }
    llm = configure_model()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory
    )
    tools = [
        Tool(
            name="ESG QA Tool",
            func=qa_chain.invoke,
            description="Answers user queries specifically related to the ESG policy document.",
        )
    ]
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent="conversational-react-description",
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
        agent_kwargs={
            "system_message": (
                "You are a document-based QA assistant with access to an ESG policy document. Your goal is to help users by answering their questions using only the information found in the document. Maintain context during conversations, clarify ambiguities, and avoid speculation."
                "Handle follow-up and unrelated questions appropriately. Provide concise, relevant answers."
                "If a query is unsafe or harmful, respond with: 'I'm sorry, I can't help with that request.'\n"
            )
        },
    )
    response = agent_executor.run(query)
    docs = retriever.get_relevant_documents(query)
    scores = evaluate_answer(
        llm, query, response, [d.page_content for d in docs], embedding
    )

    return response, scores
