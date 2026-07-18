import os 
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd
import spacy 

#loading a pretrained model
nlp = spacy.load("en_core_sci_sm")

#using pretrained model to identify keywords from medical docs
def extract_entities(text):
    doc = nlp(text)

    entities = []

    for ent in doc.ents:
        entities.append(ent.text.lower().strip())
    return list(set(entities))

#classification of main entities envolved in medical terms:
def classify_entities(entities):
    
    metadata = {
        "symptoms" : [],
        "conditions" :[],
        "medications" :[],
        "treatments" :[]
    }

    for entity in entities:

        if entity in SYMPTOMS:
            metadata["symptoms"].append(entity)
        elif entity in CONDITIONS:
            metadata["conditions"].append(entity)
        elif entity in MEDICATIONS:
            metadata["medications"].append(entity) 
        elif entity in TREATMENTS:
            metadata["treatments"].append(entity)    

#dictionary for the classification done:
SYMPTOMS = {
    "headache", "dizziness", "fatigue", "cough", "chest pain",
    "fever", "nausea", "vomiting", "shortness of breath",
    "abdominal pain", "back pain", "joint pain", "rash",
    "itching", "swelling", "weight loss", "weight gain",
    "blurred vision", "hearing loss", "palpitations",
    "diarrhea", "constipation", "loss of appetite",
    "night sweats", "fainting", "weakness", "confusion",
    "anxiety", "depression", "insomnia", "tremor",
    "seizures", "sore throat", "runny nose", "nasal congestion"
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
    "lymphoma"
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
    "doxycycline"
}


documents = []


pdf_folder = "docs"

#embeddding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


for file in os.listdir(pdf_folder):
    texts=[]
    if file.endswith(".pdf"):
        path = os.path.join(pdf_folder, file)

        loader = PyPDFLoader(path)
        docs = loader.load()
        metadatas=[]

        #metadata for keeping sources at the time of output
        for doc in docs:
            
            doc.metadata["file_name"] = file
            doc.metadata["page_number"] = doc.metadata["page"] + 1 # in page attribute of meta data we already have page num but it starts form zero.

            entities = extract_entities(doc.page_content)

            classified = classify_entities(entities)

            doc.metadatas.append({
                # New metadata
                "symptoms": classified["symptoms"],
                "conditions": classified["conditions"],
                "medications": classified["medications"],
                "treatments": classified["treatments"]
            })

        documents.extend(docs)#adds each page one by one in documents , here if we add multiple documents they are kinda stacked over one another
print(len(documents))

#function for page by page chunking
def chunk_page(text, chunk_size=500, chunk_overlap=50):
    chunks=[]
    start=0

    while(start<len(text)):
        end = start + chunk_size
        chunk = text[start:end]

        if(end<len(text)):
            last_period=chunk.rfind(".")
            
            if last_period > chunk_size*0.8:
                chunk = chunk[:last_period+1] # +1 in order to include the full stop 
                end = start + last_period + 1
            
        chunks.append(chunk)
        
        start = end - chunk_overlap # resetting the start value to overlap a part of previous chunk
    
    return chunks

def add_in_batches(collection, ids, documents, metadatas, embeddings, batch_size=5000):
    """
    Chroma's add() rejects any single call bigger than its internal max batch size, which is 5461.
    Therefore, adding 5000 every time
    """
    total = len(ids)
    for start in range(0, total, batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end]
        )


client = chromadb.PersistentClient(path="./chroma_db")

guidelines_collection = client.get_or_create_collection(
    name="ml_chunks"
)
qa_collection = client.get_or_create_collection(
    name="qa_chunks"
)

qa_id = 0  #global id counter for csv's to keep ids unique

for file in os.listdir(pdf_folder):
    if not file.endswith(".csv"):
        continue

    texts = []
    metadatas = []

    csv_path = os.path.join(pdf_folder, file)
    df = pd.read_csv(csv_path)

    for row_idx, row in df.iterrows():
        # one chunk per row -> question+answer pair stays together, never mixed with another row
        text = f"""
    Question : {row['question']}
    Answer : {row['answer']}
    """.strip()

    texts.append(text)

    entities = extract_entities(text)

    classified = classify_entities(entities)

    metadatas.append({
        "source": str(row["source"]),
        "focus_area": str(row["focus_area"]),
        "type": "qa",
        "file_name": file,
        "row_number": int(row_idx) + 1,

        # New metadata
        "symptoms": classified["symptoms"],
        "conditions": classified["conditions"],
        "medications": classified["medications"],
        "treatments": classified["treatments"]
    })

    if not texts:
        # csv had no rows (or wrong column names) - nothing to embed
        print(f"{file}: no valid rows found, skipping")
        continue

    ids = [f"qa_{qa_id + i}" for i in range(len(texts))]
    qa_id += len(texts)

    embeddings = embedding_model.encode(texts)
    add_in_batches(
        qa_collection,
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )

    print(f"{file}: added {len(texts)} qa chunks")

chunk_id=0

for doc in documents:

    text = doc.page_content.strip()

    # Skip empty pages
    if not text:
        continue

    chunks = [
        c.strip()
        for c in chunk_page(
            text,
            chunk_size=500,
            chunk_overlap=50
        )
        if len(c.strip()) > 30
    ]

    # Skip pages that produce no valid chunks
    if not chunks:
        continue

    print(
        f"{doc.metadata['file_name']} | "
        f"page={doc.metadata['page_number']} | "
        f"chunks={len(chunks)}"
    )

    embeddings = embedding_model.encode(chunks)

    ids = [f"chunk_{chunk_id+i}" for i in range(len(chunks))]

    metadatas = [
        {
            "file_name": doc.metadata["file_name"],
            "page_number": doc.metadata["page_number"]
        }
        for _ in chunks
    ]

    add_in_batches(
        guidelines_collection,
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )

    chunk_id += len(chunks)

print(qa_collection.count())