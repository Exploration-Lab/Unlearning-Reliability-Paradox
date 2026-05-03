import math
import json
import re
import math
from collections import defaultdict

from lmi_compute import build_lmi

from nltk.corpus import stopwords


# =========================
# Clean attribution tokens
# =========================

def clean_token(tok):
    tok = tok.replace("\u0120", " ")   # space
    tok = tok.replace("\u010a", "\n")  # newline
    return tok.strip().lower()


# =========================
# Get LMI value
# =========================

def token_lmi(token, label, lmi_dict):
    return lmi_dict.get(token, {}).get(label, 0.0)


# =========================
# Extract attribution tokens
# =========================

def get_attr_tokens(example):
    attr = example["token attributions"]

    cleaned = []

    for tok, score in attr:

        if score is None or (isinstance(score, float) and math.isnan(score)):
            continue

        tok_clean = clean_token(tok)

        if tok_clean == "":
            continue

        cleaned.append((tok_clean, float(score)))

    return cleaned

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
# Shortcut tokens
# =========================

def shortcut_tokens(example, lmi_dict):

    q_tokens, a_tokens, label = extract_tokens_for_lmi(example)

    if q_tokens is None:
        return []

    attr_tokens = get_attr_tokens(example)

    results = []

    for tok, attr in attr_tokens:

        # only consider question tokens
        if tok not in q_tokens:
            continue

        lmi_val = token_lmi(tok, label, lmi_dict)

        in_ans = tok in a_tokens

        results.append({
            "token": tok,
            "attribution": attr,
            "lmi": lmi_val,
            "in_answer": in_ans
        })

    return results

def filter_shortcuts(tokens, attr_thresh=0.01, lmi_thresh=1e-4):

    return [
        t for t in tokens
        if abs(t["attribution"]) >= attr_thresh
        and t["lmi"] >= lmi_thresh
    ]

def compute_shortcuts(data, lmi_dict):

    all_results = []
    
    stop_words = set(stopwords.words('english')) - set(['what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how'])

        
    psc = 0
    for ex in data:

        tokens = shortcut_tokens(ex, lmi_dict)

        top_attr_tokens = sorted(tokens, key=lambda x: abs(x["attribution"]), reverse=True)[:10]
        attr_thresh = top_attr_tokens[-1]["attribution"] if len(top_attr_tokens) > 0 else 0.0
        shortcuts = filter_shortcuts(tokens, attr_thresh=attr_thresh, )
        shortcuts = [t for t in shortcuts if t["token"].lower() in stop_words]

        all_results.append({
            "prompt": ex["prompt"],
            "answer": ex["answer"],
            "shortcut_tokens": shortcuts
        })
        
        if len(shortcuts) > 0:
            psc += 1
        
    psc = psc / len(data)
    print(f"PSC={psc:.1%}")
    return all_results

def save_shortcuts(data, lmi_dict, output_path="debug_shortcuts.json"):

    results = compute_shortcuts(data, lmi_dict)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # print("Saved shortcut tokens to", output_path)
    

if __name__ == "__main__":
    base_path = "/DATA3/divyaksh/Projects/inseq_testing/ig_attribution/outputs/"
    paths = [
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_integrated_gradients_mcq_data__mcqa__forget01_shuffle/attributions.json", "Pretrained forget"),
        
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_integrated_gradients_mcq_data__mcqa__retain99_shuffle/attributions.json", "Pretrained retain"),
        
        (f"{base_path}open-unlearning__tofu_Llama-3.1-8B-Instruct_full_integrated_gradients_mcq_data__mcqa__forget01_shuffle/attributions.json", "Full Finetuned forget"),
        
        (f"{base_path}open-unlearning__tofu_Llama-3.1-8B-Instruct_full_integrated_gradients_mcq_data__mcqa__retain99_shuffle/attributions.json", "Full Finetuned retain"),
        
        (f"{base_path}open-unlearning__tofu_Llama-3.1-8B-Instruct_retain99_integrated_gradients_mcq_data__mcqa__forget01_shuffle/attributions.json", "Retained 1% forget"),
        (f"{base_path}open-unlearning__tofu_Llama-3.1-8B-Instruct_retain95_integrated_gradients_mcq_data__mcqa__forget05_shuffle/attributions.json", "Retained 5% forget"),
        (f"{base_path}open-unlearning__tofu_Llama-3.1-8B-Instruct_retain90_integrated_gradients_mcq_data__mcqa__forget10_shuffle/attributions.json", "Retained 10% forget"),
        
        (f"{base_path}open-unlearning__tofu_Llama-3.1-8B-Instruct_retain99_integrated_gradients_mcq_data__mcqa__retain99_shuffle/attributions.json", "Retained 99% retain"),
        (f"{base_path}open-unlearning__tofu_Llama-3.1-8B-Instruct_retain95_integrated_gradients_mcq_data__mcqa__retain95_shuffle/attributions.json", "Retained 95% retain"),
        (f"{base_path}open-unlearning__tofu_Llama-3.1-8B-Instruct_retain90_integrated_gradients_mcq_data__mcqa__retain90_shuffle/attributions.json", "Retained 90% retain"),
        
        
        
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_GradAscent_integrated_gradients_mcq_data__mcqa__forget01_shuffle/attributions.json", "GradAscent 1% forget"),
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_GradAscent_integrated_gradients_mcq_data__mcqa__forget05_shuffle/attributions.json", "GradAscent 5% forget"),
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_GradAscent_integrated_gradients_mcq_data__mcqa__forget10_shuffle/attributions.json", "GradAscent 10% forget"),
        
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_GradAscent_integrated_gradients_mcq_data__mcqa__retain99_shuffle/attributions.json", "GradAscent 99% retain"),
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_GradAscent_integrated_gradients_mcq_data__mcqa__retain95_shuffle/attributions.json", "GradAscent 95% retain"),
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_GradAscent_integrated_gradients_mcq_data__mcqa__retain90_shuffle/attributions.json", "GradAscent 90% retain"),
        
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_GradDiff_integrated_gradients_mcq_data__mcqa__forget01_shuffle/attributions.json", "GradDiff 1% forget"),
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_GradDiff_integrated_gradients_mcq_data__mcqa__forget05_shuffle/attributions.json", "GradDiff 5% forget"),
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_GradDiff_integrated_gradients_mcq_data__mcqa__forget10_shuffle/attributions.json", "GradDiff 10% forget"),
        
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_GradDiff_integrated_gradients_mcq_data__mcqa__retain99_shuffle/attributions.json", "GradDiff 99% retain"),
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_GradDiff_integrated_gradients_mcq_data__mcqa__retain95_shuffle/attributions.json", "GradDiff 95% retain"),
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_GradDiff_integrated_gradients_mcq_data__mcqa__retain90_shuffle/attributions.json", "GradDiff 90% retain"),
        
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_NPO_integrated_gradients_mcq_data__mcqa__forget01_shuffle/attributions.json", "NPO 1% forget"),
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_NPO_integrated_gradients_mcq_data__mcqa__forget05_shuffle/attributions.json", "NPO 5% forget"),
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_NPO_integrated_gradients_mcq_data__mcqa__forget10_shuffle/attributions.json", "NPO 10% forget"),
        
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_NPO_integrated_gradients_mcq_data__mcqa__retain99_shuffle/attributions.json", "NPO 99% retain"),
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_NPO_integrated_gradients_mcq_data__mcqa__retain95_shuffle/attributions.json", "NPO 95% retain"),
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_NPO_integrated_gradients_mcq_data__mcqa__retain90_shuffle/attributions.json", "NPO 90% retain"),
        
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_DPO_integrated_gradients_mcq_data__mcqa__forget01_shuffle/attributions.json", "DPO 1% forget"),
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_DPO_integrated_gradients_mcq_data__mcqa__forget05_shuffle/attributions.json", "DPO 5% forget"),
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_DPO_integrated_gradients_mcq_data__mcqa__forget10_shuffle/attributions.json", "DPO 10% forget"),
        
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_DPO_integrated_gradients_mcq_data__mcqa__retain99_shuffle/attributions.json", "DPO 99% retain"),
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_DPO_integrated_gradients_mcq_data__mcqa__retain95_shuffle/attributions.json", "DPO 95% retain"),
        (f"{base_path}meta-llama__Llama-3.1-8B-Instruct_DPO_integrated_gradients_mcq_data__mcqa__retain90_shuffle/attributions.json", "DPO 90% retain"),
    ]
    # input_path = "/DATA3/divyaksh/Projects/inseq_testing/ig_attribution/outputs/meta-llama__Llama-3.1-8B-Instruct_GradAscent_integrated_gradients_mcq_data__mcqa__forget01_shuffle/attributions.json"
    
    for input_path, desc in paths:
        print(f"...{desc}...")

        with open(input_path, "r") as f:
            data = json.load(f)

        
        lmi_dict = build_lmi(input_path)
            
        with open("top_5_percent_non_stop_tokens.txt", "r") as f:
            top_tokens = set(line.strip() for line in f)
            
        

        save_shortcuts(data, lmi_dict)