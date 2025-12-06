from sentence_transformers import SentenceTransformer, util
import torch
from config import CACHE_DIR

model = SentenceTransformer("BAAI/bge-large-en-v1.5", cache_folder=CACHE_DIR).to("cuda")

def gen_embeddings(texts):
    return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True).to("cuda")

def least_similar(texts, embeddings):
    text_embeddings = gen_embeddings(texts)
    cosine_scores = util.cos_sim(text_embeddings, embeddings)
    return torch.argmin(cosine_scores, dim=1), cosine_scores

def median_similar(texts, embeddings):
    text_embeddings = gen_embeddings(texts)
    cosine_scores = util.cos_sim(text_embeddings, embeddings)
    return torch.median(cosine_scores, dim=1)[1], cosine_scores

def most_similar(texts, embeddings):
    text_embeddings = gen_embeddings(texts)
    cosine_scores = util.cos_sim(text_embeddings, embeddings)
    return torch.argmax(cosine_scores, dim=1), cosine_scores

def most_similar_cluster(texts, embeddings, clusters):
    text_embeddings = gen_embeddings(texts)
    cosine_scores = util.cos_sim(text_embeddings, embeddings)
    best_clusters, best_scores = [-1] * len(texts), [-1] * len(texts)

    for cluster, facts in clusters.items():
        scores = torch.mean(cosine_scores[:,facts], dim=1)
        for i in range(len(texts)):
            if scores[i].item() > best_scores[i]:
                best_clusters[i] = cluster
                best_scores[i] = scores[i].item()
    
    return best_clusters, cosine_scores

def self_sim(texts):
    embeddings = gen_embeddings(texts)
    return util.cos_sim(embeddings, embeddings)