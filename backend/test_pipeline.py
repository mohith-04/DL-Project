import time
from services.retriever import retrieve_per_source
from services.classifier import classify_sources
from services.synthesizer import synthesize_answer

sid = "ab396256-1dd3-4ad6-a165-bc671c11ac85"
q = "What is RAG?"

t0 = time.time()
print("1. Retrieving...")
chunks = retrieve_per_source(sid, q, top_k=3)
print(f"Retrieved {len(chunks)} sources in {time.time()-t0:.2f}s")

t1 = time.time()
print("2. Classifying...")
cls = classify_sources(q, chunks)
print(f"Classified: {cls['classification']} in {time.time()-t1:.2f}s")

t2 = time.time()
print("3. Synthesizing...")
ans = synthesize_answer(q, chunks, cls)
print(f"Synthesized: {len(ans)} chars in {time.time()-t2:.2f}s")
