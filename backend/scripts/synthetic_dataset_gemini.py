import pymongo
import json
import hashlib
import uuid
import os
import time
import google.generativeai as genai

# ==========================================
# CONFIGURATION
# ==========================================
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "legal_ai_research"     
COLLECTION_NAME = "chunks"        

# Apni actual Gemini API key daalein
GEMINI_API_KEY = "" 

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-3.1-flash-lite')

EXPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "exports", "synthetic_qa_dataset.json")

def generate_multilingual_qa(chunk_text):
    """
    Ek legal text chunk analyze karta hai. Agar chunk suitable nahi hai, toh skip karta hai.
    Nahi toh 3 languages me questions aur ek English reference answer generate karta hai.
    """
    prompt = f"""
    Analyze the following legal text chunk. 
    Determine if it contains enough clear, factual, and complete information to form a basic, straightforward question.
    
    CRITICAL RULE: If the text is too fragmented, lacks context, is just a heading/index, or is NOT suitable for creating a good question, you MUST output EXACTLY and ONLY the word: SKIP
    
    Otherwise, if it IS suitable, generate a single specific question in 3 different languages/scripts, and provide a single reference answer in English.
    Format the output EXACTLY like this:
    Question_EN: [Question in pure English]
    Question_HI: [Same question in Hindi script (Devanagari)]
    Question_HINGLISH: [Same question in Hinglish (Hindi words written using English alphabet)]
    Answer: [Your clear reference answer in English]
    
    Text: {chunk_text}
    """
    
    try:
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        # Agar Gemini ne decide kiya ki chunk suitable nahi hai
        if result.upper().startswith("SKIP") or "SKIP" in result[:10]:
            return "UNSUITABLE", ""
            
        # Agar suitable hai toh Text ko parse karenge
        q_en = result.split("Question_EN:")[1].split("Question_HI:")[0].strip()
        q_hi = result.split("Question_HI:")[1].split("Question_HINGLISH:")[0].strip()
        q_hinglish = result.split("Question_HINGLISH:")[1].split("Answer:")[0].strip()
        ans = result.split("Answer:")[1].strip()
        
        return {"en": q_en, "hi": q_hi, "hinglish": q_hinglish}, ans
        
    except Exception as e:
        print(f"Gemini parsing error (Might not have followed format): {e}")
        return "ERROR", ""

def main():
    client_mongo = pymongo.MongoClient(MONGO_URI)
    db = client_mongo[DB_NAME]
    collection = db[COLLECTION_NAME]

    dataset = []
    total_docs = collection.count_documents({})
    print(f"Found {total_docs} total documents in MongoDB ({DB_NAME}.{COLLECTION_NAME}).")
    
    if total_docs == 0:
        print("Error: Collection khaali hai.")
        return

    # Ab hum thode zyada chunks fetch kar rahe hain, taaki agar kuch skip ho jaye, tab bhi humare paas data aaye
    print("Fetching up to 15 chunks (some might be skipped)...")
    chunks = collection.find().limit(15) 
    
    successful_chunks = 0
    
    for chunk in chunks:
        # Jab tak 5 successful chunks na mil jayein, loop chalne do (optional limit)
        if successful_chunks >= 5:
            break
            
        chunk_text = chunk.get("text", "") 
        doc_id = chunk.get("document_id", f"DOC_{uuid.uuid4().hex[:16].upper()}")
        chunk_id = chunk.get("chunk_id", f"{doc_id}_C{str(uuid.uuid4().hex[:6]).upper()}")
        title = chunk.get("title", "Constitution of India")
        article = chunk.get("article", "Unknown Article")
        
        if not chunk_text or len(chunk_text.split()) < 10: # 10 words se chote chunks wese hi skip kar do
            continue
            
        print(f"\nEvaluating chunk: {chunk_id}...")
        chunk_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
        
        # QA Generate karo ya Skip check karo
        result_dict, answer = generate_multilingual_qa(chunk_text)
        
        # Handle SKIP
        if result_dict == "UNSUITABLE":
            print(f"⏭️  SKIPPED: Chunk {chunk_id} does not have enough context for a basic question.")
            time.sleep(1) # API limit ke liye thoda rukna
            continue
            
        # Handle Error
        if result_dict == "ERROR" or not answer:
            print(f"⚠️  ERROR: Failed to generate QA format for {chunk_id}.")
            time.sleep(1)
            continue
            
        print(f"✅ SUCCESS: Generated QA for chunk {chunk_id}.")
        successful_chunks += 1
        group_id = f"SYN_{uuid.uuid4().hex[:6].upper()}"
        
        for lang, q_text in result_dict.items():
            question_id = f"{group_id}_{lang.upper()}"
            record = {
                "question_id": question_id,
                "group_id": group_id,
                "language": lang,
                "domain": "constitutional_law",
                "difficulty": "easy",
                "question": q_text,
                "reference_answer": answer,
                "gold_document_ids": [doc_id], 
                "gold_chunk_ids": [chunk_id],  
                "gold_chunk_hashes": {chunk_id: chunk_hash}, 
                "gold_authorities": [{
                    "document_id": doc_id,
                    "chunk_ids": [chunk_id],
                    "title": title,
                    "article": article,
                    "version": "current"
                }],
                "answerable_from_kb": True,
                "requires_multiple_sources": False,
                "notes": f"Human approved synthetic candidate. Language: {lang}"
            }
            dataset.append(record)
            
        time.sleep(2) 

    if dataset:
        os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)
        with open(EXPORT_PATH, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"\nSuccessfully generated {len(dataset)} records in JSON format.")
        print(f"Dataset saved to: {EXPORT_PATH}")
    else:
        print("\nNo records were generated. All chunks were skipped or failed.")

if __name__ == "__main__":
    main()