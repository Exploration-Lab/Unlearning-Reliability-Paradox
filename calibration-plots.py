
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

CHOICES = ["A", "B", "C", "D"]


def load_data(path, key="forget_mcqa_prob"):
    with open(path, "r") as f:
        data = json.load(f)

    values = data[key]["value_by_index"]

    y_true = []
    y_pred = []
    probs = []

    for k in values:
        item = values[k]

        y_true.append(item["label"])
        y_pred.append(item["generated choice"])
        probs.append(item["prob"])

    probs = np.array(probs)

    return y_true, y_pred, probs


# -----------------------
# Accuracy + F1
# -----------------------

def compute_accuracy_f1(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, f1


# -----------------------
# Brier score (multiclass)
# -----------------------

def compute_brier(y_true, probs):

    y_true_idx = np.array([CHOICES.index(y) for y in y_true])

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y_true_idx)), y_true_idx] = 1

    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))

    return brier


# -----------------------
# ECE / MCE
# -----------------------

def compute_ece_mce(y_true, probs, n_bins):

    y_true_idx = np.array([CHOICES.index(y) for y in y_true])

    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)

    correct = (pred == y_true_idx).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    mce = 0.0

    for i in range(n_bins):

        mask = (conf >= bins[i]) & (conf < bins[i + 1])

        if np.sum(mask) == 0:
            continue

        acc_bin = np.mean(correct[mask])
        conf_bin = np.mean(conf[mask])

        gap = abs(acc_bin - conf_bin)

        ece += np.sum(mask) / len(conf) * gap
        mce = max(mce, gap)

    return ece, mce

# -----------------------
# Reliability Diagram
# -----------------------


def plot_reliability_diagram(y_true, probs, n_bins, title="", key=""):

    y_true_idx = np.array([CHOICES.index(y) for y in y_true])

    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)

    correct = (pred == y_true_idx).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)

    bin_acc = []
    bin_conf = []
    bin_count = []

    for i in range(n_bins):

        mask = (conf >= bins[i]) & (conf < bins[i + 1])

        if np.sum(mask) == 0:
            bin_acc.append(0)
            bin_conf.append(0)
            bin_count.append(0)
            continue

        bin_acc.append(np.mean(correct[mask]))
        bin_conf.append(np.mean(conf[mask]))
        bin_count.append(np.sum(mask))

    bin_acc = np.array(bin_acc)
    bin_conf = np.array(bin_conf)
    bin_count = np.array(bin_count)

    # ----- plot -----

    # fig = plt.figure()

    # bars = accuracy
    plt.bar(
        np.linspace(0.05, 0.95, n_bins),
        bin_acc,
        width=0.08*10/n_bins,
        alpha=0.6,
        label="Accuracy",
    )

    # confidence line
    plt.plot(
        np.linspace(0.05, 0.95, n_bins),
        bin_conf,
        marker="o",
        label="Confidence",
    )

    # perfect calibration
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"{title} - {key}", fontsize=20)

    # return fig
    # plt.savefig(f"saves/eval/plots/phi-1_5/{title}_{key}_reliability_diagram.pdf", dpi=300, bbox_inches="tight")
    # plt.show()
    


# -----------------------
# MAIN
# -----------------------

def print_metrics_and_plot(path, title, subfig_nrows=None, subfig_ncols=None, subfig_pos=None, n_bins=20):
    
    ret_values = []
    
    forget_percentage = int(title.split()[-1].replace("%",""))
    retain_percentage = 100 - forget_percentage
    title = " ".join(title.split()[:-1])
    
    for i, (key, key_title) in enumerate([
        ("retain_mcqa_prob", f"Retain {retain_percentage:2.0f}%"),
        ("forget_mcqa_prob", f"Forget {forget_percentage:2.0f} %") 
        ]):
        print(f"--- {key_title} ---")
        y_true, y_pred, probs = load_data(path, key=key)

        acc, f1 = compute_accuracy_f1(y_true, y_pred)
        brier = compute_brier(y_true, probs)
        ece, mce = compute_ece_mce(y_true, probs, n_bins=n_bins)

        print("Accuracy:", acc)
        print("F1:", f1)
        print("Brier:", brier)
        print("ECE:", ece)
        print("MCE:", mce)
        
        if subfig_nrows is not None and subfig_ncols is not None and subfig_pos is not None:
            plt.subplot(subfig_nrows, subfig_ncols, subfig_pos+i)
        plot_reliability_diagram(y_true, probs, n_bins=n_bins, title=title, key=key_title)
        
        ret_values.append({"key": key, "accuracy": acc, "f1": f1, "brier": brier, "ece": ece, "mce": mce})
        
    return ret_values

###############################################################################


path_template = "saves/eval/{task_name}/RELU_EVAL.json"
unlearn_path_template = "saves/unlearn/{task_name}/{checkpoint}/evals/RELU_EVAL.json"
tasks = [    
    ("relu_Llama-3.1-8B-Instruct_pretrained_retain99", "Pretrained 1%"),
    ("relu_Llama-3.1-8B-Instruct_full_retain99", "Full Finetuned 1%"),
    ("relu_Llama-3.1-8B-Instruct_retain99", "Retained 1%"),
]

unlearn_tasks = [
    # ("tofu_Llama-3.1-8B-Instruct_forget01_GradAscent", "checkpoint-2", "GradAscent 1%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget01_GradAscent", "checkpoint-5", "GradAscent 1%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget01_GradAscent", "checkpoint-7", "GradAscent 1%"),
    ("tofu_Llama-3.1-8B-Instruct_forget01_GradAscent", "checkpoint-10", "GradAscent 1%"),
    
    # ("tofu_Llama-3.1-8B-Instruct_forget01_GradDiff", "checkpoint-2", "GradDiff 1%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget01_GradDiff", "checkpoint-5", "GradDiff 1%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget01_GradDiff", "checkpoint-7", "GradDiff 1%"),
    ("tofu_Llama-3.1-8B-Instruct_forget01_GradDiff", "checkpoint-10", "GradDiff 1%"),
    
    # ("tofu_Llama-3.1-8B-Instruct_forget01_NPO", "checkpoint-2", "NPO 1%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget01_NPO", "checkpoint-5", "NPO 1%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget01_NPO", "checkpoint-7", "NPO 1%"),
    ("tofu_Llama-3.1-8B-Instruct_forget01_NPO", "checkpoint-10", "NPO 1%"),
    
    # ("tofu_Llama-3.1-8B-Instruct_forget01_DPO", "checkpoint-2", "DPO 1%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget01_DPO", "checkpoint-5", "DPO 1%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget01_DPO", "checkpoint-7", "DPO 1%"),
    ("tofu_Llama-3.1-8B-Instruct_forget01_DPO", "checkpoint-10", "DPO 1%"),
]

nrows = 4
ncols = 4
n_bins = 100
fig = plt.figure(figsize=(ncols*5, nrows*4))

results = []

for i, (task, task_name) in enumerate(tasks):
    try:
        print(f"=== {task} ===")
        metrics = print_metrics_and_plot(path_template.format(task_name=task), title=task_name, subfig_nrows=nrows, subfig_ncols=ncols, subfig_pos=i*4+1, n_bins=n_bins)
        for m in metrics:
            results.append({"task": task, "checkpoint": 0, **m})
    except Exception as e:
        print(f"Error processing {task}: {e}")
        raise e

for i, (task, checkpoint, method) in enumerate(unlearn_tasks):
    try:
        print(f"=== {task} | {checkpoint} ===")
        metrics = print_metrics_and_plot(unlearn_path_template.format(task_name=task, checkpoint=checkpoint), title=method, subfig_nrows=nrows, subfig_ncols=ncols, subfig_pos=(i*4)+2+1, n_bins=n_bins)
        for m in metrics:
            results.append({"task": task, "checkpoint": checkpoint, **m})
    except Exception as e:
        print(f"Error processing {task}|{checkpoint}: {e}")
        raise e

handles, labels = plt.gca().get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='center',
    bbox_to_anchor=(0.25, 0.12),
    fontsize=18*2,
    frameon=True,
    markerscale=3,
    handlelength=3,
)
plt.tight_layout()
plt.savefig(f"saves/eval/plots/Llama-3.1-8B-Instruct_forget01_reliability_diagram_bins_{n_bins}.pdf", dpi=300, bbox_inches="tight")

df = pd.DataFrame(results)
# store all the results in a JSON file
with open(f"saves/eval/calibration_results_only_8B_forget01_bins_{n_bins}.json", "w") as f:
    json.dump(results, f, indent=4)
    


###############################################################################

path_template = "saves/eval/{task_name}/RELU_EVAL.json"
unlearn_path_template = "saves/unlearn/{task_name}/{checkpoint}/evals/RELU_EVAL.json"
tasks = [    
    ("relu_Llama-3.1-8B-Instruct_pretrained_retain95", "Pretrained 5%"),
    ("relu_Llama-3.1-8B-Instruct_full_retain95", "Full Finetuned 5%"),
    ("relu_Llama-3.1-8B-Instruct_retain95", "Retained 5%"),
]

unlearn_tasks = [
    # ("tofu_Llama-3.1-8B-Instruct_forget05_GradAscent", "checkpoint-12", "GradAscent 5%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget05_GradAscent", "checkpoint-25", "GradAscent 5%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget05_GradAscent", "checkpoint-37", "GradAscent 5%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget05_GradAscent", "checkpoint-50", "GradAscent 5%"),
    ("tofu_Llama-3.1-8B-Instruct_forget05_GradAscent", "checkpoint-60", "GradAscent 5%"),
    
    # ("tofu_Llama-3.1-8B-Instruct_forget05_GradDiff", "checkpoint-12", "GradDiff 5%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget05_GradDiff", "checkpoint-25", "GradDiff 5%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget05_GradDiff", "checkpoint-37", "GradDiff 5%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget05_GradDiff", "checkpoint-50", "GradDiff 5%"),
    ("tofu_Llama-3.1-8B-Instruct_forget05_GradDiff", "checkpoint-60", "GradDiff 5%"),
    
    # ("tofu_Llama-3.1-8B-Instruct_forget05_NPO", "checkpoint-12", "NPO 5%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget05_NPO", "checkpoint-25", "NPO 5%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget05_NPO", "checkpoint-37", "NPO 5%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget05_NPO", "checkpoint-50", "NPO 5%"),
    ("tofu_Llama-3.1-8B-Instruct_forget05_NPO", "checkpoint-60", "NPO 5%"),
    
    # ("tofu_Llama-3.1-8B-Instruct_forget05_DPO", "checkpoint-12", "DPO 5%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget05_DPO", "checkpoint-25", "DPO 5%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget05_DPO", "checkpoint-37", "DPO 5%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget05_DPO", "checkpoint-50", "DPO 5%"),
    ("tofu_Llama-3.1-8B-Instruct_forget05_DPO", "checkpoint-60", "DPO 5%"),
]

nrows = 4
ncols = 4
fig = plt.figure(figsize=(ncols*5, nrows*4))

results = []

for i, (task, task_name) in enumerate(tasks):
    try:
        print(f"=== {task} ===")
        metrics = print_metrics_and_plot(path_template.format(task_name=task), title=task_name, subfig_nrows=nrows, subfig_ncols=ncols, subfig_pos=i*4+1, n_bins=n_bins)
        for m in metrics:
            results.append({"task": task, "checkpoint": 0, **m})
    except Exception as e:
        print(f"Error processing {task}: {e}")
        raise e

for i, (task, checkpoint, method) in enumerate(unlearn_tasks):
    try:
        print(f"=== {task} | {checkpoint} ===")
        metrics = print_metrics_and_plot(unlearn_path_template.format(task_name=task, checkpoint=checkpoint), title=method, subfig_nrows=nrows, subfig_ncols=ncols, subfig_pos=(i*4)+2+1, n_bins=n_bins)
        for m in metrics:
            results.append({"task": task, "checkpoint": checkpoint, **m})
    except Exception as e:
        print(f"Error processing {task}|{checkpoint}: {e}")
        raise e

handles, labels = plt.gca().get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='center',
    bbox_to_anchor=(0.25, 0.12),
    fontsize=18*2,
    frameon=True,
    markerscale=3,
    handlelength=3,
)
plt.tight_layout()
plt.savefig(f"saves/eval/plots/Llama-3.1-8B-Instruct_forget05_reliability_diagram_bins_{n_bins}.pdf", dpi=300, bbox_inches="tight")
df = pd.DataFrame(results)
# store all the results in a JSON file
with open(f"saves/eval/calibration_results_only_8B_forget05_bins_{n_bins}.json", "w") as f:
    json.dump(results, f, indent=4)

###############################################################################

path_template = "saves/eval/{task_name}/RELU_EVAL.json"
unlearn_path_template = "saves/unlearn/{task_name}/{checkpoint}/evals/RELU_EVAL.json"
tasks = [
    ("relu_Llama-3.1-8B-Instruct_pretrained_retain90", "Pretrained 10%"),
    ("relu_Llama-3.1-8B-Instruct_full_retain90", "Full Finetuned 10%"),
    ("relu_Llama-3.1-8B-Instruct_retain90", "Retained 10%"),
]

unlearn_tasks = [
    # ("tofu_Llama-3.1-8B-Instruct_forget10_GradAscent", "checkpoint-25", "GradAscent 10%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget10_GradAscent", "checkpoint-50", "GradAscent 10%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget10_GradAscent", "checkpoint-75", "GradAscent 10%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget10_GradAscent", "checkpoint-100", "GradAscent 10%"),
    ("tofu_Llama-3.1-8B-Instruct_forget10_GradAscent", "checkpoint-125", "GradAscent 10%"),
    
    # ("tofu_Llama-3.1-8B-Instruct_forget10_GradDiff", "checkpoint-25", "GradDiff 10%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget10_GradDiff", "checkpoint-50", "GradDiff 10%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget10_GradDiff", "checkpoint-75", "GradDiff 10%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget10_GradDiff", "checkpoint-100", "GradDiff 10%"),
    ("tofu_Llama-3.1-8B-Instruct_forget10_GradDiff", "checkpoint-125", "GradDiff 10%"),
    
    # ("tofu_Llama-3.1-8B-Instruct_forget10_NPO", "checkpoint-25", "NPO 10%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget10_NPO", "checkpoint-50", "NPO 10%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget10_NPO", "checkpoint-75", "NPO 10%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget10_NPO", "checkpoint-100", "NPO 10%"),
    ("tofu_Llama-3.1-8B-Instruct_forget10_NPO", "checkpoint-125", "NPO 10%"),
    
    # ("tofu_Llama-3.1-8B-Instruct_forget10_DPO", "checkpoint-25", "DPO 10%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget10_DPO", "checkpoint-50", "DPO 10%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget10_DPO", "checkpoint-75", "DPO 10%"),
    # ("tofu_Llama-3.1-8B-Instruct_forget10_DPO", "checkpoint-100", "DPO 10%"),
    ("tofu_Llama-3.1-8B-Instruct_forget10_DPO", "checkpoint-125", "DPO 10%"),
]

nrows = 4
ncols = 4
fig = plt.figure(figsize=(ncols*5, nrows*4))

results = []

for i, (task, task_name) in enumerate(tasks):
    try:
        print(f"=== {task} ===")
        metrics = print_metrics_and_plot(path_template.format(task_name=task), title=task_name, subfig_nrows=nrows, subfig_ncols=ncols, subfig_pos=i*4+1, n_bins=n_bins)
        for m in metrics:
            results.append({"task": task, "checkpoint": 0, **m})
    except Exception as e:
        print(f"Error processing {task}: {e}")
        raise e

for i, (task, checkpoint, method) in enumerate(unlearn_tasks):
    try:
        print(f"=== {task} | {checkpoint} ===")
        metrics = print_metrics_and_plot(unlearn_path_template.format(task_name=task, checkpoint=checkpoint), title=method, subfig_nrows=nrows, subfig_ncols=ncols, subfig_pos=(i*4)+2+1, n_bins=n_bins)
        for m in metrics:
            results.append({"task": task, "checkpoint": checkpoint, **m})
    except Exception as e:
        print(f"Error processing {task}|{checkpoint}: {e}")
        raise e

handles, labels = plt.gca().get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='center',
    bbox_to_anchor=(0.25, 0.12),
    fontsize=18*2,
    frameon=True,
    markerscale=3,
    handlelength=3,
)
plt.tight_layout()
plt.savefig(f"saves/eval/plots/Llama-3.1-8B-Instruct_forget10_reliability_diagram_bins_{n_bins}.pdf", dpi=300, bbox_inches="tight")

df = pd.DataFrame(results)
# store all the results in a JSON file
with open(f"saves/eval/calibration_results_only_8B_forget10_bins_{n_bins}.json", "w") as f:
    json.dump(results, f, indent=4)





