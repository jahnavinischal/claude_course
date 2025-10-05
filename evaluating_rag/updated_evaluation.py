# imports
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import json

# initialize models 
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Example RAG data (can be replaced dynamically)
query = "What is the capital of France?"
retrieved_context = "France is a country in Europe. Its capital city is Paris."
rag_response = "The capital of France is Paris."

# can be replaced by below while evaluating on a given RAG data:

# rag_data = {
#     "query": user_input,
#     "context": retrieved_docs,
#     "response": model_output
# }
# result = evaluate_rag_response(rag_data)


# semantic similarity metrics 

def cosine_sim(a, b):
    emb_a = embed_model.encode(a, convert_to_tensor=True)
    emb_b = embed_model.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb_a, emb_b).item()

faithfulness = cosine_sim(rag_response, retrieved_context)
relevance = cosine_sim(rag_response, query)

 # fluency/perplexity metric 
def compute_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        loss = gpt2_model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()

perplexity = compute_perplexity(rag_response)
fluency = 1 / perplexity  # invert since lower perplexity is better

# normalize & combine 
# normalize fluency roughly between 0-1
fluency_score = min(1.0, fluency * 10)

# weighted composite score 
final_score = round((0.4 * faithfulness + 0.4 * relevance + 0.2 * fluency_score), 3)

# output 

evaluation = {
    "faithfulness": round(faithfulness, 3),
    "relevance": round(relevance, 3),
    "fluency": round(fluency_score, 3),
    "final_score": final_score,
}

print(json.dumps(evaluation, indent=2))


# output:
# {
#   "faithfulness": 0.904,
#   "relevance": 0.879,
#   "fluency": 0.224,
#   "final_score": 0.758
# }

# these scores are semantic and reference-free 
# higher = better response quality
