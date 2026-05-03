import json
import re
import math
from collections import defaultdict


# =========================
# Parse example
# =========================

def parse_example(example):
    prompt = example["prompt"]
    answer = example["answer"]

    # Extract label (A/B/C/D)
    match = re.search(r"Answer:\s*([A-D])", answer)
    if match is None:
        return None, None, None

    label = match.group(1)

    lines = prompt.split("\n")

    question_lines = []
    options = {}

    for line in lines:
        line = line.strip()

        # option line
        if re.match(r"^[A-D]\.", line):
            key = line[0]
            options[key] = line[2:].strip()

        # question line
        elif line and not line.startswith("Answer"):
            question_lines.append(line)

    question_text = " ".join(question_lines)

    if label not in options:
        return None, None, None

    answer_text = options[label]

    return question_text, answer_text, label


# =========================
# Tokenize
# =========================

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# =========================
# Extract tokens for LMI
# =========================

def extract_tokens_for_lmi(example):

    q_text, a_text, label = parse_example(example)

    if q_text is None:
        return None, None, None

    q_tokens = set(tokenize(q_text))
    a_tokens = set(tokenize(a_text))

    return q_tokens, a_tokens, label


# =========================
# Build counts
# =========================

def compute_counts(data):

    word_count = defaultdict(int)
    label_count = defaultdict(int)
    joint_count = defaultdict(lambda: defaultdict(int))

    N = len(data)

    for ex in data:

        q_tokens, a_tokens, label = extract_tokens_for_lmi(ex)

        if q_tokens is None:
            continue

        label_count[label] += 1

        for w in q_tokens:
            word_count[w] += 1
            joint_count[w][label] += 1

    return word_count, label_count, joint_count, N


# =========================
# Compute LMI
# =========================

def compute_lmi(word_count, label_count, joint_count, N):

    lmi = defaultdict(dict)

    for w in word_count:

        for y in label_count:

            c = joint_count[w][y]

            if c == 0:
                continue

            p_wy = c / N
            p_w = word_count[w] / N
            p_y = label_count[y] / N
            p_y_given_w = c / word_count[w]

            lmi[w][y] = p_wy * math.log(
                (p_y_given_w / p_y) + 1e-12
            )

    return lmi


# =========================
# Main
# =========================

def build_lmi(input_path, output_path):

    with open(input_path, "r") as f:
        data = json.load(f)

    word_count, label_count, joint_count, N = compute_counts(data)

    lmi = compute_lmi(
        word_count,
        label_count,
        joint_count,
        N
    )

    with open(output_path, "w") as f:
        json.dump(lmi, f, indent=2)

    print("Saved LMI to", output_path)


# =========================
# RUN
# =========================

if __name__ == "__main__":

    input_json = "data.json"     # <-- change
    output_json = "lmi.json"

    build_lmi(input_json, output_json)