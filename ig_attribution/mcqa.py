import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class MCQADataset(Dataset):
    def __init__(
        self,
        json_path,
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
        self.json_path = json_path
        self.data = self.load_hf_dataset_from_json()
        self.add_dataset_index()
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
        
    def load_hf_dataset_from_json(self):
        # Load the dataset from the JSON file
        dataset = load_dataset("json", data_files=self.json_path)
        return dataset["train"]
    
    def add_dataset_index(self):
        indexing = np.arange(len(self.data))
        self.data = self.data.add_column("index", indexing)

    def __len__(self):
        return 40
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

        # answer = item[self.answer_key]
        answer = item[item[self.correct_option_key][0]]

        item_dct = {
            "question": question,
            "choices": options,
            "answer": answer,
        }
        
        return item_dct