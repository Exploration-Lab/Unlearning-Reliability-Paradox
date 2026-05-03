import re

def extract_label(answer_text):
    match = re.search(r"Answer:\s*([A-D])", answer_text)
    return match.group(1)

import re

def tokenize(text):
    # lowercase + basic word split
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens

from collections import defaultdict

def compute_counts(data):
    word_count = defaultdict(int)
    label_count = defaultdict(int)
    joint_count = defaultdict(lambda: defaultdict(int))

    N = len(data)

    for ex in data:
        label = extract_label(ex["answer"])
        label_count[label] += 1

        tokens = set(tokenize(ex["prompt"]))  # set → presence, not frequency

        for w in tokens:
            word_count[w] += 1
            joint_count[w][label] += 1

    return word_count, label_count, joint_count, N

import math

def compute_lmi(word_count, label_count, joint_count, N):
    lmi = defaultdict(dict)

    for w in word_count:
        for y in label_count:
            c_wy = joint_count[w][y]

            if c_wy == 0:
                continue

            p_wy = c_wy / N
            p_w = word_count[w] / N
            p_y = label_count[y] / N
            p_y_given_w = c_wy / word_count[w]

            lmi[w][y] = p_wy * math.log(p_y_given_w / p_y + 1e-12)

    return lmi

import json

def build_lmi(input_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    word_count, label_count, joint_count, N = compute_counts(data)
    lmi = compute_lmi(word_count, label_count, joint_count, N)

    return lmi

if __name__ == "__main__":
    input_path = "/DATA3/divyaksh/Projects/inseq_testing/ig_attribution/outputs/meta-llama__Llama-3.1-8B-Instruct_GradAscent_integrated_gradients_mcq_data__mcqa__forget01_shuffle/attributions.json"
    lmi = build_lmi(input_path)

    with open("lmi_scores.json", "w") as f:
        json.dump(lmi, f, indent=2)