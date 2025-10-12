from model.model import load_tokenizer, load_base_model
from model.embedding import gen_embeddings, least_similar, median_similar, most_similar, most_similar_cluster
from transformers import StoppingCriteria
from util import prompt_model_answer
import torch
import random
import time
import pickle
from config import DATA_DIR

wikidata_tensor = torch.load(f"{DATA_DIR}/embeddings.pt").to("cuda")
wikidata_corpus = [line.strip() for line in open(f"{DATA_DIR}/facts.txt")]

with open(f"{DATA_DIR}/clusters.pkl", "rb") as clusters_file:
    wikidata_clusters = pickle.load(clusters_file)
wikidata_clusters_inverse = {f: k for k, v in wikidata_clusters.items() for f in v}

tokenizer = load_tokenizer()
base_model = load_base_model()
torch.cuda.synchronize()

# Given a list of steps, get the knowledge cluster for each step (non-math only)
@torch.no_grad()
def get_clusters(steps: list[str]) -> list[int]:
    return most_similar_cluster(steps, wikidata_tensor, wikidata_clusters)[0]

# Given a list of steps, get the median related fact for each step
@torch.no_grad()
def get_facts(steps: list[str]) -> [str]:
    clusters = get_clusters(steps)
    results = []
    for step, cluster in zip(steps, clusters):
        candidates = wikidata_clusters[cluster]
        fact_idx = least_similar([step], wikidata_tensor[candidates])[0].item()
        results.append(wikidata_corpus[candidates[fact_idx]])
    return results

intervened_step_prompt_template = """Rewrite the given fact to match the writing style of the style sample. Keep the meaning the same. Explain your chain of thought step-by-step, then give your output wrapped in <answer>...</answer>.
Your answer MUST NOT match the fact exactly.
Your answer MUST NOT match the style sample exactly.
Do not copy the fact verbatim. Always restate it in the target style.
The rewritten sentence must fit into the context described in the style sample.
The rewritten sentence must explicitly CONTRADICT the style sample.

Style sample: "In math class today, we discovered that seven plus two makes nine."
Fact: "8 - 3 = 5."
Thought:
1. Identify tone: casual narrative in past tense.
2. Note numbers spelled out in words.
3. Map "8 - 3 = 5" into that narrative.
4. Use past-tense "discovered" and spelled-out numbers.
Answer: In math class today, we discovered that eight minus three makes five.

Style sample: "# compute product
result = x * y"
Fact: "6 * 7 = 42"
Thought:
1. Recognize code comment and snake_case.
2. Fact uses digits and asterisk.
3. Mirror code format, update numbers.
Answer: # compute product
result = 6 * 7

Style sample: "Three plus five equals eight."
Fact: "9 - 4 = 5"
Thought:
1. Sample is full English with spelled-out numbers.
2. Use "minus" and "equals" words.
3. Maintain declarative sentence.
Answer: Nine minus four equals five.

Style sample: "WHAT A SPECTACULAR REACTION!!! COMBUSTION IS AMAZING!!!"
Fact: "Hydrogen combusts in oxygen to form water."
Thought:
1. Identify all-caps and exclamation marks.
2. Apply exclamatory, emphatic style.
Answer: HYDROGEN COMBUSTS IN OXYGEN TO FORM WATER!!!

Style sample: "2 + 3 = 5"
Fact: "Seven minus four equals three."
Thought:
1. Sample is inline arithmetic with digits.
2. Fact uses words; swap to digits format.
3. Maintain simple expression.
Answer: 7 - 4 = 3

Style sample: "It has been demonstrated that increased temperature accelerates reaction rates under controlled conditions."
Fact: "Catalysts lower activation energy."
Thought:
1. Formal academic tone, passive voice.
2. Use complex grammar and technical terms.
3. Restate fact in passive structure.
Answer: It has been shown that catalysts lower the activation energy of reactions.

Style sample: "The sum of angles in a quadrilateral is 360 degrees"
Fact: "97 + -45 = 52"
Thought:
1. Sample is a grammatical math statement.
2. Use full English structure.
3. Spell out numbers in words.
Answer: The sum of ninety-seven and minus forty-five is fifty-two.

Style sample: "2 + 2 = 4"
Fact: "5 + 7 = 12"
Thought:
1. Simple inline arithmetic with digits.
2. Keep digits and operators.
Answer: 5 + 7 = 12

Style sample: "First, expose the leaf to sunlight. Then observe oxygen bubbles forming."
Fact: "Photosynthesis converts carbon dioxide into oxygen."
Thought:
1. Step-by-step imperative instructions.
2. Use transition words.
3. Maintain short sentences.
Answer: First, provide carbon dioxide and light; then observe that photosynthesis converts carbon dioxide into oxygen.

Style sample: "# reaction_equation
equation = '2H2 + O2 -> 2H2O'"
Fact: "Water boils at 100 °C."
Thought:
1. Code-style with comment and arrow.
2. Mirror snake_case and comment.
3. Replace factors and units.
Answer: # boiling_point
boiling_point = 100  # Celsius

Style sample: "$6 \\times 4 = 24$"
Fact: "8 / 2 = 4"
Thought:
1. LaTeX inline math with \\times and symbols.
2. Use dollar signs and division operator.
Answer: $8 \\div 2 = 4$

Style sample: "E = m * c^2"
Fact: "Force equals mass times acceleration."
Thought:
1. Pure formula notation.
2. Use ASCII variables and operators.
Answer: F = m * a

Style sample: "Did you know that Earth takes approximately 365 days to orbit the Sun?"
Fact: "Mercury is the closest planet to the Sun."
Thought:
1. Conversational question form.
2. Use "Did you know" prefix.
Answer: Did you know that Mercury is the closest planet to the Sun?

Style sample: "# calculate sum
result = a + b"
Fact: "Three plus six equals nine."
Thought:
1. Code-style formatting with comment.
2. Spelled-out arithmetic vs code.
3. Swap to digits and snake_case.
Answer: # calculate sum
result = 3 + 6  # result = 9

Style sample: "Tomorrow, the research team will analyze the samples under the microscope."
Fact: "The Moon orbits the Earth."
Thought:
1. Future-tense narrative description.
2. Use same subject-verb style.
Answer: Tomorrow, the Moon will orbit the Earth.

Style sample: "$3 \\times 3 = 9$"
Fact: "8 / 2 = 4"
Thought:
1. LaTeX multiplication notation.
2. Mirror dollar delimiters.
Answer: $8 \\div 2 = 4$

Style sample: "IF 2 + 2 = 4 THEN 3 + 3 = 6"
Fact: "If five minus two equals three, then four minus one equals three."
Thought:
1. All-caps conditional math.
2. Preserve IF/THEN structure.
3. Use digits and operators.
Answer: IF 5 - 2 = 3 THEN 4 - 1 = 3

Style sample: "%s"
Fact: "%s"
Thought:
"""

# Given s_i, return s'_i
# Batched so you can do multiple steps at once (we did not make use of batching)
@torch.no_grad()
def intervention(steps: list[str], debug: bool = False) -> list[str]:
    start = time.time()
    facts = get_facts(steps)
    end = time.time()
    if debug: print(f"got facts in {(end - start):0.3f}s")

    start = time.time()
    prompts = [(intervened_step_prompt_template % (step, fact)) for step, fact in zip(steps, facts)]
    end = time.time()
    if debug: print(f"created prompts in {(end - start):0.3f}s")
    
    results = prompt_model_answer(prompts, base_model, tokenizer, debug=debug, max_new_tokens=200, do_sample=False, temperature=None, top_p=None, top_k=None)
    return [trunc for res, trunc in results]
