import numpy as np
import torch
from torch.utils.data import Dataset

from data.utils import load_hf_dataset, preprocess_chat_instance, add_dataset_index, load_hf_dataset_from_json


class MCQADataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key="question",
        answer_key="answer",
        option_keys=["A", "B", "C", "D"],
        correct_option_key="Correct option",
        few_shot_dataset_hf_args=None,
        max_length=512,
        predict_with_generate=False,
        shuffle_options=True, 
        n_options=4, 
        seed=42
    ):
        super(MCQADataset, self).__init__()
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.data = load_hf_dataset(**hf_args)
        # self.json_path = hf_args["data_files"]["train"] if "train" in hf_args["data_files"] else list(hf_args["data_files"].values())[0]
        self.json_path = hf_args["json_path"]
        self.data = load_hf_dataset_from_json(self.json_path)
        self.data = add_dataset_index(self.data)
        self.shuffle_options = shuffle_options
        self.n_options = n_options
        self.template_args = template_args
        self.question_key = question_key
        self.answer_key = answer_key
        self.option_keys = option_keys
        self.correct_option_key = correct_option_key
        self.predict_with_generate = predict_with_generate
        self.max_length = max_length
        self.seed = seed
        np.random.seed(seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        index = self.data[idx]["index"] if "index" in self.data[idx] else idx
        question = item[self.question_key]
        try:
            options = [item[key] for key in self.option_keys]
        except:
            raise ValueError(
                f"Options not found for question {question} in dataset {self.dataset_path}"
            )

        correct_option = item[self.correct_option_key][0]
        answer = item[self.answer_key]
        answer = item[item[self.correct_option_key][0]]
        prompt, label = self.format_prompt(question, options, answer, correct_option)
        
        tokenizer_data = preprocess_chat_instance(
            self.tokenizer,
            self.template_args,
            [prompt],
            [label],
            self.max_length,
            self.predict_with_generate,
        )
        # Shifting the labels by 2 tokens to the right to account for the added generation prompt in the chat template
        tokenizer_data["labels"] = torch.tensor(tokenizer_data["labels"][2:])
        
        # Change all the lables to -100 except the last 2 tokens which is the correct option label for loss calculation
        tokenizer_data["labels"][:-2] = -100

        item_dct = {
            "input_ids": tokenizer_data["input_ids"],
            "labels": tokenizer_data["labels"],
            "attention_mask": tokenizer_data["attention_mask"],
            "index": index,
        }
        
        return item_dct

    def format_prompt(self, question, options, answer, correct_option):
        try:
            correct_option = options.index(answer)
        except ValueError:
            raise ValueError(
                f"Answer {answer} not found in options {options} for question {question} in dataset {self.dataset_path}"
            )
        # correct_option = ord(correct_option) - 65

        wrong_options = [i for i in range(self.n_options) if i != correct_option]

        # sample wrong options based on n_options
        wrong_options = np.random.choice(
            wrong_options, self.n_options - 1, replace=False
        )
        options = [options[correct_option]] + [
            options[wrong_option] for wrong_option in wrong_options
        ]

        if self.shuffle_options:
            np.random.shuffle(options)

        correct_option = options.index(answer)

        prompt_template_text = f"Question: {question} \n"
        for i, option in enumerate(options):
            # A. option1 \n B. option2 \n C. option3 \n D. option4 \n
            prompt_template_text += f"{chr(65 + i)}. {option}\n"
        # prompt_template_text += "Answer:"

        label = " " + chr(65 + correct_option)

        return prompt_template_text, label