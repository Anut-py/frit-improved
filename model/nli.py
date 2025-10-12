from transformers import pipeline

nli = pipeline("text-classification", model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")

premise_template = """
Question: %s

Answer 1: "%s"

Answer 2: "%s"
"""

hypothesis = "Answer 1 and Answer 2 are exactly the same in the context of the question."

def answers_equivalent_nli(q, a, b, threshold=0.95):
    premise = premise_template % (q, a, b)
    result = nli({"text": premise, "text_pair": hypothesis})
    return result["label"] == "entailment" and result["score"] > threshold