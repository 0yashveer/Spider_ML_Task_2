import sys
import json
import chromadb
from sentence_transformers import SentenceTransformer

from ingest_fixed import(
    nlp, extract_entities, classify_entities,
    entities_to_chromadb, extract_abbreviations
)

CHROMA_PATH = "./chroma_db"
GUIDELINES_COLLECTION = "ml_chunks"
QA_COLLECTION = "qa_chunks"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3:8b"
OLLAMA_HOST = "http://localhost:11434"

SEMANTIC_TOP_K=15
FINAL_TOP_N=3


def analyze_query(query):

    entities = extract_entities(query)
    classified = classify_entities(entities)
    abbreviations = extract_abbreviations(query)
    fields = entities_to_chromadb(entities, classified)

    return entities, classified, fields

def where_filter(classified):
    metadataCount=[]

    if classified["symptoms"]:
        metadataCount.append({"symptoms_count":{"$gt":0}})
    if classified["conditions"]:
        metadataCount.append({"conditions_count":{"$gt":0}})
    if classified["medications"]:
        metadataCount.append({"medications_count": {"$gt": 0}})
    if classified["treatments"]:
        metadataCount.append({"treatments_count": {"$gt": 0}})

    if not metadataCount:
        return None
    
    if len(metadataCount)==1:
        return metadataCount[0]

    return {"$or":metadataCount}


def hard_filter(collection, query_embedding, where_clause, top_k=15):

    if where_clause is None:
        return None

    result = collection.query(
        query_embeddings=[query_embedding],
        where=where_clause,
        n_results=top_k
    )
    return result

def fallback_semantic_search(collection, query_embedding, top_k):

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

def retrieve_candidates(collection, query_embedding, classified, top_k=SEMANTIC_TOP_K):
    where_clause = where_filter(classified)

    hard_result = hard_filter(collection, query_embedding, where_clause, top_k)
    if hard_result:
        return hard_result, False

    fallback_result = fallback_semantic_search(collection, query_embedding, top_k)
    return fallback_result, True


def metadata_overlap_score(query_classified, chunk_metadata):
    score=0
    for category in ("symptoms", "conditions", "medications", "treatments"):
        query_label = [label for label in query_classified[category]]
        chunk_field = (chunk_metadata.get(category) or "").lower()
        for label in query_label:
            if label and label in chunk_field:
                score+=1
    return score

def rerank(result, query_classified, final_top_n=FINAL_TOP_N):

    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    scored=[]
    for chunk_id, doc, meta, distance in zip(ids, documents, metadatas, distances):
        overlap = metadata_overlap_score(query_classified, meta)

        similarity = 1.0/(1.0 + distance)

        combined = (similarity*0.6) + (overlap/4)*0.4
        scored.append({
            "id":chunk_id,
            "text":doc,
            "metadata":meta,
            "distance":distance,
            "overlap":overlap,
            "combined_score":combined
        })

    scored.sort(key=lambda x:x["combined_score"], reverse=True)
    return scored[:final_top_n]

def context(top_chunks):
    final=[]
    for i, chunk in enumerate(top_chunks, start=1):
        meta = chunk["metadata"]
        if meta.get("type") == "guidelines":
            source_dest = f"{meta.get('file_name')}, page {meta.get('page_number')}"
        else:
            source_dest = f"{meta.get('file_name')}, row {meta.get('row_num')} (source:{meta.get('source')})"

        final.append(
            f"Source{i} : {source_dest}\n"
            f"Conditions : {meta.get('conditions')} | "
            f"Symptoms : {meta.get('symptoms')} | "
            f"Medications : {meta.get('medications')} | "
            f"Treatments : {meta.get('treatments')}\n"
            f"Content : {chunk['text']}\n\n"
        )
    return final

def format_history(history):
    if not history:
        return ""
    lines = []
    for turn in history:
        speaker = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {turn['content']}")
    return "\n".join(lines)


def prompt_builder(user_query, top_chunks, history=None):
    context_block = context(top_chunks)
    history_block = format_history(history)

    history_section = f"""
PREVIOUS CONVERSATION (for context only — do not treat as a source of medical facts):
{history_block}
""" if history_block else ""

    prompt = f"""You are a medical information assistant. Answer the user's question using ONLY the
context sources below.

Rules:
- Keep your answer short: 2-4 sentences maximum, unless the question genuinely requires a list.
- Do not quote sources verbatim or at length. Paraphrase in your own words.
- Cite sources briefly by their [Source N] label inline, not with long quotations.
- If the sources don't contain enough information to answer confidently, say so in one sentence instead of guessing.
- Use the previous conversation only to understand what the user is referring to (e.g. "it", "that", "instead") — never pull facts from it that aren't in the content sources below.
- Do not give a diagnosis. End with a brief reminder to consult a healthcare professional only if the topic is medically significant (skip it for trivial/definitional questions).
{history_section}
CONTENT SOURCES: {context_block}

USER_QUESTION: {user_query}

ANSWER (short, 2-4 sentences):"""
    return prompt

def compute_confidence(ranked_chunks, used_fallback, classified):
    if not ranked_chunks:
        return 0.0

    top = ranked_chunks[0]
    top_similarity = 1.0 / (1.0 + top["distance"])
    top_overlap_ratio = top["overlap"] / 4

    base = (top_similarity * 0.6) + (top_overlap_ratio * 0.4)

    if used_fallback:
        base *= 0.85

    entity_count = sum(len(v) for v in classified.values())
    if entity_count == 0:
        base *= 0.85

    if len(ranked_chunks) > 1:
        gap = top["combined_score"] - ranked_chunks[1]["combined_score"]
        if gap < 0.05:
            base *= 0.9

    return round(min(max(base, 0.0), 1.0), 3)

def answer_query(user_query, collection_name=QA_COLLECTION, history=None):
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=collection_name)

    entities, classified, query_field = analyze_query(user_query)
    print("Query entities detected:")
    print(f"  symptoms : {classified['symptoms']}")
    print(f"  conditions : {classified['conditions']}")
    print(f"  medications : {classified['medications']}")
    print(f"  treatments : {classified['treatments']}")

    query_embedding = embedding_model.encode(user_query).tolist()
    result, used_fallback = retrieve_candidates(collection, query_embedding, classified)

    if not result['ids'][0]:
        return {
            "answer": "No relevant info found in knowledge base for this.",
            "confidence": 0.0,
            "sources": [],
            "detected_entities": classified
        }

    ranked_chunks = rerank(result, classified)
    confidence = compute_confidence(ranked_chunks, used_fallback, classified)
    prompt = prompt_builder(user_query, ranked_chunks, history = history)
    answer = ollama_call(prompt)

    return {
        "answer": answer or "The model did not return a response.",
        "confidence": confidence,
        "sources": [
            {
                "metadata": c["metadata"],
                "score": round(c["combined_score"], 3),
                "text": c["text"]
            }
            for c in ranked_chunks
        ],
        "detected_entities": classified
    }


def ollama_call(prompt, model=OLLAMA_MODEL, host=OLLAMA_HOST):
    try:
        import ollama
        client = ollama.Client(host=host)
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"unable to load the model: {type(e).__name__}: {e}")
        return None