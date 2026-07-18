"""
query_pipeline.py

Retrieval + generation companion to ingest_fixed.py.

Pipeline:
  1. Take the user query and run it through the SAME entity extraction /
     classification used at ingestion time (extract_entities, classify_entities).
     This gives the query its own symptoms/conditions/medications/treatments,
     exactly like every chunk in the DB has.
  2. HARD METADATA FILTER (primary strategy): use those extracted labels to
     build a Chroma `where` filter (using the *_count fields and $contains-style
     matching via the comma-joined string fields) so semantic search only runs
     over chunks that mention at least one of the same medical concepts.
  3. FALLBACK: if the hard filter returns too few candidates (e.g. the query
     has no recognized entities, or the filter is too narrow), fall back to
     plain semantic search over the whole collection so the user never gets
     zero results.
  4. Within whichever candidate set was selected, run embedding similarity
     search (Chroma's own query) to rank by semantic closeness.
  5. Re-rank the top semantic hits by metadata overlap with the query, so a
     chunk that matches both semantically AND on overlapping conditions/symptoms
     floats to the top over one that's only semantically similar.
  6. Pass the final top-N chunks to a local LLM via Ollama, with their text +
     metadata as grounded context, and ask it to answer using only that context.

Run:
    python query_pipeline.py "what should I take for a headache with high blood pressure"

Requires:
    pip install chromadb sentence-transformers spacy scispacy ollama --break-system-packages
    ollama pull llama3.1   (or whatever OLLAMA_MODEL is set to below)
    `ollama serve` running locally (default http://localhost:11434)
"""

import sys
import json
import chromadb
from sentence_transformers import SentenceTransformer

# Reuse everything from your ingestion script instead of duplicating logic.
# This guarantees the query is processed with the exact same entity
# extraction / classification pipeline used when the chunks were indexed.
from ingest_fixed import (
    nlp,
    extract_entities,
    extract_abbreviations,
    classify_entities,
    entities_to_chromadb,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHROMA_PATH = "./chroma_db"
GUIDELINES_COLLECTION = "ml_chunks"
QA_COLLECTION = "qa_chunks"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

OLLAMA_MODEL = "llama3.1"          # change to whatever you've pulled locally
OLLAMA_HOST = "http://localhost:11434"

MIN_HARD_FILTER_RESULTS = 3        # if hard filter returns fewer than this, fall back
SEMANTIC_TOP_K = 20                # how many candidates to pull from Chroma before rerank
FINAL_TOP_N = 5                    # how many chunks actually get sent to the LLM


# ---------------------------------------------------------------------------
# Step 1: query -> entities -> metadata (reuses ingestion functions directly)
# ---------------------------------------------------------------------------

def analyze_query(query_text):
    """
    Run the query through the same pipeline used on chunks at ingestion time.
    Returns (entities, classified, chroma_style_fields).
    """
    entities = extract_entities(query_text)
    classified = classify_entities(entities)
    abbreviations = extract_abbreviations(query_text)
    fields = entities_to_chromadb(entities, classified)
    fields["abbreviations"] = json.dumps(abbreviations) if abbreviations else ""
    return entities, classified, fields


# ---------------------------------------------------------------------------
# Step 2: build a hard metadata filter from the query's classification
# ---------------------------------------------------------------------------

def build_where_filter(classified):
    """
    Build a Chroma `where` clause that matches any chunk whose metadata
    string fields contain at least one of the query's extracted labels.

    Chroma's `where` filtering doesn't support substring search on string
    fields directly (only exact match / numeric comparisons / $in on exact
    values), so instead of trying to do $contains tricks we build an $or of
    exact-field-nonempty checks per category that the query actually hit.
    This is intentionally a coarse "did we detect this category at all"
    filter -- the precise overlap scoring happens later in step 5 (rerank),
    where we do real substring/set comparison in Python.

    If you want category-level hard filtering (e.g. "only look at chunks
    that have at least one condition AND at least one symptom"), this is
    where to express that.
    """
    clauses = []

    if classified["symptoms"]:
        clauses.append({"symptoms_count": {"$gt": 0}})
    if classified["conditions"]:
        clauses.append({"conditions_count": {"$gt": 0}})
    if classified["medications"]:
        clauses.append({"medications_count": {"$gt": 0}})
    if classified["treatments"]:
        clauses.append({"treatments_count": {"$gt": 0}})

    if not clauses:
        return None  # nothing recognized in the query -> no filter, go straight to fallback

    if len(clauses) == 1:
        return clauses[0]

    return {"$or": clauses}


# ---------------------------------------------------------------------------
# Step 2b / 3: hard-filtered search with fallback to plain semantic search
# ---------------------------------------------------------------------------

def hard_filtered_search(collection, query_embedding, where_filter, top_k):
    """
    Primary strategy: semantic search restricted to chunks matching where_filter.
    Returns the raw Chroma query result, or None if the filter was empty/too narrow.
    """
    if where_filter is None:
        return None

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
    )

    num_hits = len(result["ids"][0]) if result["ids"] else 0
    if num_hits < MIN_HARD_FILTER_RESULTS:
        return None  # not enough to be useful -> caller falls back

    return result


def fallback_semantic_search(collection, query_embedding, top_k):
    """
    Fallback strategy: plain semantic search over the whole collection,
    no metadata filter at all.
    """
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )


def retrieve_candidates(collection, query_embedding, classified, top_k=SEMANTIC_TOP_K):
    """
    Implements: try hard filter first; if it doesn't return enough, fall back
    to full semantic search. Returns (result, used_fallback: bool).
    """
    where_filter = build_where_filter(classified)

    hard_result = hard_filtered_search(collection, query_embedding, where_filter, top_k)
    if hard_result is not None:
        return hard_result, False

    fallback_result = fallback_semantic_search(collection, query_embedding, top_k)
    return fallback_result, True


# ---------------------------------------------------------------------------
# Step 5: rerank candidates by metadata overlap with the query
# ---------------------------------------------------------------------------

def metadata_overlap_score(query_classified, chunk_metadata):
    """
    Counts how many of the query's symptom/condition/medication/treatment
    labels also appear in the chunk's corresponding comma-joined metadata
    string. Simple, transparent overlap score used to break ties and boost
    chunks that match on medical concept, not just embedding distance.
    """
    score = 0
    for category in ("symptoms", "conditions", "medications", "treatments"):
        query_labels = [label.lower() for label in query_classified[category]]
        chunk_field = (chunk_metadata.get(category) or "").lower()
        for label in query_labels:
            if label and label in chunk_field:
                score += 1
    return score


def rerank(result, query_classified, final_top_n=FINAL_TOP_N):
    """
    Takes a raw Chroma query result (single query, so index [0] throughout),
    combines embedding distance with metadata overlap, and returns the
    final_top_n best chunks as a list of dicts.
    """
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    scored = []
    for chunk_id, doc, meta, distance in zip(ids, documents, metadatas, distances):
        overlap = metadata_overlap_score(query_classified, meta)
        # Lower distance = more similar. We turn it into a similarity score
        # so higher = better, consistently with overlap.
        similarity = 1.0 / (1.0 + distance)
        # Weighted combination: semantic similarity matters most, metadata
        # overlap acts as a boost. Tune these weights as needed.
        combined = (0.7 * similarity) + (0.3 * (overlap / 4.0))  # /4 since 4 categories max
        scored.append({
            "id": chunk_id,
            "text": doc,
            "metadata": meta,
            "distance": distance,
            "overlap": overlap,
            "combined_score": combined,
        })

    scored.sort(key=lambda x: x["combined_score"], reverse=True)
    return scored[:final_top_n]


# ---------------------------------------------------------------------------
# Step 6: build a grounded prompt and call Ollama
# ---------------------------------------------------------------------------

def build_context_block(ranked_chunks):
    """
    Formats the final chunks into a context block for the LLM prompt,
    including source info so the model can (and should) cite sources.
    """
    parts = []
    for i, chunk in enumerate(ranked_chunks, start=1):
        meta = chunk["metadata"]
        if meta.get("type") == "guidelines":
            source_desc = f"{meta.get('file_name')}, page {meta.get('page_number')}"
        else:
            source_desc = f"{meta.get('file_name')}, row {meta.get('row_num')} (source: {meta.get('source')})"

        parts.append(
            f"[Source {i}] ({source_desc})\n"
            f"Conditions: {meta.get('conditions') or 'none'} | "
            f"Symptoms: {meta.get('symptoms') or 'none'} | "
            f"Medications: {meta.get('medications') or 'none'} | "
            f"Treatments: {meta.get('treatments') or 'none'}\n"
            f"Content: {chunk['text']}\n"
        )
    return "\n".join(parts)


def build_prompt(user_query, ranked_chunks):
    context_block = build_context_block(ranked_chunks)
    prompt = f"""You are a medical information assistant. Answer the user's question using ONLY the
context sources below. If the sources don't contain enough information to answer
confidently, say so explicitly instead of guessing. Cite sources by their [Source N]
label when you use them. Do not give a diagnosis - present the retrieved information
and recommend the user consult a healthcare professional for personal medical advice.

CONTEXT SOURCES:
{context_block}

USER QUESTION:
{user_query}

ANSWER:"""
    return prompt


def call_ollama(prompt, model=OLLAMA_MODEL, host=OLLAMA_HOST):
    """
    Calls a local Ollama instance. Uses the `ollama` python package if
    installed; otherwise falls back to a raw HTTP request so this script
    works even without the extra dependency.
    """
    try:
        import ollama
        client = ollama.Client(host=host)
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
    except ImportError:
        import urllib.request

        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        return data["message"]["content"]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def answer_query(user_query, collection_name=QA_COLLECTION):
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=collection_name)

    # Step 1
    entities, classified, query_fields = analyze_query(user_query)
    print("Query entities detected:")
    print(f"  symptoms:    {classified['symptoms']}")
    print(f"  conditions:  {classified['conditions']}")
    print(f"  medications: {classified['medications']}")
    print(f"  treatments:  {classified['treatments']}")

    # Step 2/3 (embed the query itself for the semantic part)
    query_embedding = embedding_model.encode(user_query).tolist()
    result, used_fallback = retrieve_candidates(collection, query_embedding, classified)
    print(f"\nRetrieval strategy used: {'FALLBACK (semantic only)' if used_fallback else 'HARD METADATA FILTER'}")

    if not result["ids"][0]:
        return "No relevant information found in the knowledge base for this query.", []

    # Step 5
    ranked_chunks = rerank(result, classified)

    # Step 6
    prompt = build_prompt(user_query, ranked_chunks)
    answer = call_ollama(prompt)

    return answer, ranked_chunks


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python query_pipeline.py "your question here"')
        sys.exit(1)

    user_query = " ".join(sys.argv[1:])
    answer, ranked_chunks = answer_query(user_query)

    print("\n--- Top retrieved chunks ---")
    for chunk in ranked_chunks:
        print(f"id={chunk['id']} score={chunk['combined_score']:.3f} overlap={chunk['overlap']} dist={chunk['distance']:.4f}")

    print("\n--- Answer ---")
    print(answer)
