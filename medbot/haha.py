import os
import json
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import chromadb
import spacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker
from collections import defaultdict

#scispacy pipeline setup

nlp = spacy.load("en_core_sci_sm")

#inserting new elements for better embeddings in nlp pipeline
nlp.add_pipe("abbreviation_detector")
nlp.add_pipe(
    "scispacy_linker",
    config={"resolve_abbreviations":True, "linker_name":"umls"})
linker = nlp.get_pipe("scispacy_linker")

#setting a bar for threshold for linking umls concepts together and avoid rubbish linking
UMLS_LINK_THRESHOLD = 0.85


# Since we already get the tui's form our linker but the linker as no idea as which code represents which category 
# so we hardcode it as dictionaries.
SYMPTOM_TUIS = {"T184"}  # Sign or Symptom

CONDITION_TUIS = {
    "T047",  # Disease or Syndrome
    "T191",  # Neoplastic Process
    "T046",  # Pathologic Function
    "T048",  # Mental or Behavioral Dysfunction
    "T019",  # Congenital Abnormality
    "T037",  # Injury or Poisoning
}

MEDICATION_TUIS = {
    "T121",  # Pharmacologic Substance
    "T200",  # Clinical Drug
    "T195",  # Antibiotic
}

TREATMENT_TUIS = {
    "T061",  # Therapeutic or Preventive Procedure
    "T060",  # Diagnostic Procedure
    "T058",  # Health Care Activity
}

# In case the linker misses the keywords due to less confidence
# used hardcoded values for classification
SYMPTOMS = {
    "headache", "dizziness", "fatigue", "cough", "chest pain",
    "fever", "nausea", "vomiting", "shortness of breath",
    "abdominal pain", "back pain", "joint pain", "rash",
    "itching", "swelling", "weight loss", "weight gain",
    "blurred vision", "hearing loss", "palpitations",
    "diarrhea", "constipation", "loss of appetite",
    "night sweats", "fainting", "weakness", "confusion",
    "anxiety", "depression", "insomnia", "tremor",
    "seizures", "sore throat", "runny nose", "nasal congestion",
}

CONDITIONS = {
    "hypertension", "diabetes", "asthma",
    "heart disease", "coronary artery disease",
    "heart failure", "stroke",
    "chronic kidney disease", "kidney stones",
    "copd", "pneumonia", "bronchitis",
    "tuberculosis",
    "arthritis", "osteoarthritis", "rheumatoid arthritis",
    "osteoporosis",
    "anemia",
    "hypothyroidism", "hyperthyroidism",
    "obesity",
    "depression", "anxiety disorder",
    "alzheimers disease", "parkinsons disease",
    "epilepsy",
    "migraine",
    "gastroesophageal reflux disease",
    "peptic ulcer disease",
    "hepatitis",
    "cirrhosis",
    "cancer",
    "breast cancer",
    "lung cancer",
    "prostate cancer",
    "colon cancer",
    "leukemia",
    "lymphoma",
}

MEDICATIONS = {
    "amlodipine", "metformin", "aspirin",
    "lisinopril", "losartan", "atenolol",
    "hydrochlorothiazide",
    "atorvastatin", "simvastatin",
    "insulin",
    "glipizide",
    "albuterol",
    "prednisone",
    "ibuprofen",
    "acetaminophen",
    "omeprazole",
    "pantoprazole",
    "levothyroxine",
    "warfarin",
    "clopidogrel",
    "furosemide",
    "amoxicillin",
    "azithromycin",
    "doxycycline",
}

TREATMENTS = {
    "surgery", "chemotherapy", "radiation therapy", "physical therapy",
    "dialysis", "bypass surgery", "angioplasty", "vaccination",
    "biopsy", "endoscopy", "colonoscopy", "mri", "ct scan", "x-ray",
    "ultrasound", "blood transfusion", "organ transplant",
    "insulin therapy", "oxygen therapy", "counseling", "rehabilitation",
} 

#
# Entity extraction and classification
#

def extract_entities(text):

    doc = nlp(text)
    seen = set()
    entities =[]

    for ent in doc.ents:
        surface = ent.text.lower().strip()
        if not surface:
            continue

        best_cui, best_name, best_tuis, best_score = None, None, [], 0.0

        if ent._.kb_ents:
            cui, score = ent._.kb_ents[0] #ent._.kb_ents is a sorted list of (cui, score)
            if score >= UMLS_LINK_THRESHOLD:
                umls_entity = linker.kb.cui_to_entity[cui]
                best_cui = cui
                best_name = umls_entity.canonical_name
                best_tuis = list(umls_entity.types)
                best_score = float(score)
        
        dedup_key = (surface, best_cui)
        #avioding duplicaion of cui's
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        entities.append({
            "text" : surface,
            "cui": best_cui,
            "canonical_name": best_name,
            "tuis": best_tuis,
            "score": best_score
        })
    
    return entities

# for detecting abber. like BP etc
def extract_abbreviations(text):
    doc = nlp(text)
    return {
        str(abrv):str(abrv._.long_form)
        for abrv in doc._.abbreviations
    }

def classify_entities(entities):
    
    metadata = {
        "symptoms":[],
        "conditions":[],
        "medications":[],
        "treatments":[]
    }

    for entity in entities:
        label = entity["canonical_name"] or entity["text"]#when linker fails due to some spelling err or smtg else
        tuis = set(entity["tuis"])
        surface = entity["text"]

        if tuis & SYMPTOM_TUIS or surface in SYMPTOMS:
            metadata["symptoms"].append(label)
        elif tuis & CONDITION_TUIS or surface in CONDITIONS:
            metadata["conditions"].append(label)
        elif tuis & MEDICATION_TUIS or surface in MEDICATIONS:
            metadata["medications"].append(label)
        elif tuis & TREATMENT_TUIS or surface in TREATMENTS:
            metadata["treatments"].append(label)

        for key in metadata:
            metadata[key] = list(dict.fromkeys(metadata[key]))
        
    return metadata

# Chromadb cannot store dict , so converting dict to comma seperated string
def entities_to_chromadb(entities, classified):
    cuis = [e["cui"] for e in entities if e["cui"]]

    fields = {
        "symptoms" : ", ".join(classified["symptoms"]),
        "symptoms_count" : len(classified["symptoms"]),
        "conditions" : ", ".join(classified["conditions"]),
        "conditions_count" : len(classified["conditions"]),
        "medications" : ", ".join(classified["medications"]),
        "medications_count" : len(classified["medications"]),
        "treatments": ", ".join(classified["treatments"]),
        "treatments_count": len(classified["treatments"]),
        "umls_cuis": ", ".join(set(cuis)),
        "entity_count" : len(entities),
    }
    return fields

# defining lists for storing chunk_id

medical_index = {
    "cuis" : defaultdict(list),
    "symptoms" : defaultdict(list),
    "conditions" : defaultdict(list),
    "medications" : defaultdict(list),
    "treatments" : defaultdict(list)
}

#
# CHUNKING
#

def chunk_page(text, chunk_size=500, chunk_overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if end < len(text):
            last_period = chunk.rfind(".")
            if last_period > chunk_size * 0.8:
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1

        chunks.append(chunk)

        # advance start, respecting overlap, and guarantee forward progress
        next_start = end - chunk_overlap
        start = next_start if next_start > start else end

    return chunks

def add_in_batches(collection, ids, documents, metadatas, embeddings, batch_size=5000):
    total = len(ids)
    
    for start in range(0, total , batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end]
        )

def main():
    files_folder = "docs"
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path="./chroma_db")
    guidelines_collection = client.get_or_create_collection(name="ml_chunks")
    qa_collection = client.get_or_create_collection(name="qa_chunks")

    #loading pdfs into LangChain Docs 
    documents = []
    for file in os.listdir(files_folder):
        if not file.endswith(".pdf"):
            continue

        path = os.path.join(files_folder, file)
        loader = PyPDFLoader(path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["file_name"] = file
            doc.metadata["page_number"] = doc.metadata["page"] + 1
        
        documents.extend(docs)

    print(f"Loaded {len(documents)} PDF pages from '{files_folder}'")

    
    # PDFs Chunking and Embedding

    chunk_id = 0

    for doc in documents:
        text = doc.page_content.strip()
        if not text:
            continue

        raw_chunks = chunk_page(text, chunk_size=500, chunk_overlap=50)
        chunks = [c.strip() for c in raw_chunks if len(c.strip()) > 30]

        if not chunks: 
            continue

        metadatas = []
        for i, chunk in enumerate(chunks):
            current_chunk_id = f"{chunk_id+i}"
            entities = extract_entities(chunk)
            abbreviations = extract_abbreviations(chunk)
            classified = classify_entities(entities)

            for entity in entities:
                if entity["cui"]:
                    medical_index["cuis"][entity["cui"]].append(current_chunk_id)
            # ... rest of indexing ...

            for symptom in classified["symptoms"]:
                medical_index["symptoms"][symptom.lower()].append(current_chunk_id)

            for condition in classified["conditions"]:
                medical_index["conditions"][condition.lower()].append(current_chunk_id)

            for medication in classified["medications"]:
                medical_index["medications"][medication.lower()].append(current_chunk_id)

            for treatment in classified["treatments"]:
                medical_index["treatments"][treatment.lower()].append(current_chunk_id)

            meta = {
                "file_name": doc.metadata["file_name"],
                "page_number": doc.metadata["page_number"],
                "type": "guidelines",
                "abbrevations": json.dumps(abbreviations) if abbreviations else ""
            }
            meta.update(entities_to_chromadb(entities, classified))
            metadatas.append(meta)
        
        ids = [f"{chunk_id + i}" for i in range(len(chunks))]
        embeddings = embedding_model.encode(chunks)
        add_in_batches(
            guidelines_collection,
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
            embeddings=embeddings.tolist()
        )
        chunk_id += len(chunks)


    # -- CSV ingestion --

    for file in os.listdir(files_folder):
        if not file.endswith(".csv"):
            continue

        csv_path = os.path.join(files_folder, file)
        df = pd.read_csv(csv_path)

        texts = []
        metadatas = []

        for row_ids, row in df.iterrows():
            #creating one chunk per row, therefore concatinating ans and que columns
            text = f"""Question : {row['question']} 
                    Answer : {row['answer']}""".strip()
            
            entities = extract_entities(text)
            classified = classify_entities(entities)
            abbreviations = extract_abbreviations(text)

            meta = {
                "source" : str(row["source"]),
                "classified" : str(row["focus_area"]),
                "type" : "qa",
                "file_name": file,
                "row_num":int(row_ids) +1,
                "abbreviations":json.dumps(abbreviations) if abbreviations else ""
            }
            meta.update(entities_to_chromadb(entities, classified))

            texts.append(text)
            metadatas.append(meta)
        
        ids = [f"{i}" for i in range(len(texts))]

        embeddings = embedding_model.encode(texts)
        add_in_batches(
            qa_collection,
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist()
        )

        chunk_id+=len(chunks)

    print(f"guidelines collection count: {guidelines_collection.count()}")
    print(f"qa collection count: {qa_collection.count()}")

    
if __name__ == "__main__":
    main()