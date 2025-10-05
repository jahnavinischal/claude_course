from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import json

# Initialize models 
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Example RAG data (can be replaced dynamically)
query = "What is the capital of France?"
retrieved_context = "France is a country in Europe. Its capital city is Paris."
rag_response = "The capital of France is Paris."

# can be replaced by this while evaluating on a given RAG data:

# rag_data = {
#     "query": user_input,
#     "context": retrieved_docs,
#     "response": model_output
# }
# result = evaluate_rag_response(rag_data)


# Semantic Similarity Metrics 

def cosine_sim(a, b):
    emb_a = embed_model.encode(a, convert_to_tensor=True)
    emb_b = embed_model.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb_a, emb_b).item()

faithfulness = cosine_sim(rag_response, retrieved_context)
relevance = cosine_sim(rag_response, query)

# Fluency / Perplexity Metric 
def compute_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        loss = gpt2_model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()

perplexity = compute_perplexity(rag_response)
fluency = 1 / perplexity  # invert since lower perplexity = better

# Normalize & Combine 

# Normalize fluency roughly between 0-1
fluency_score = min(1.0, fluency * 10)

# Weighted composite score 
final_score = round((0.4 * faithfulness + 0.4 * relevance + 0.2 * fluency_score), 3)

# Output 

evaluation = {
    "faithfulness": round(faithfulness, 3),
    "relevance": round(relevance, 3),
    "fluency": round(fluency_score, 3),
    "final_score": final_score,
}

print(json.dumps(evaluation, indent=2))
