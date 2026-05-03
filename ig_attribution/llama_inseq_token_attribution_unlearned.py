import random
import numpy as np
import json
from pathlib import Path

import inseq
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import torch

from mcqa import MCQADataset
from tqdm import tqdm

from peft import PeftModel

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_inseq_model_tokenizer(
    model_name,
    model_path,
    lora_path=None,
    attribution_method="integrated_gradients"
):

    def _get_quantization_config():
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        # )
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,              # default, safe
            llm_int8_enable_fp32_cpu_offload=True
        )
        return bnb_config

    def _load_model_tokenizer(model_name, model_path, lora_path, bnb_config):

        # ✅ load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,              # base model
            device_map="cuda",
            quantization_config=bnb_config,
        )

        # ✅ load LoRA adapter if provided
        if lora_path is not None:
            model = PeftModel.from_pretrained(
                model,
                lora_path,
                device_map="cuda",
            )

            # optional but recommended for attribution
            model = model.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return model, tokenizer

    bnb_config = _get_quantization_config()

    model, tokenizer = _load_model_tokenizer(
        model_name,
        model_path,
        lora_path,
        bnb_config,
    )

    model = inseq.load_model(
        model,
        attribution_method,
        n_steps=5,
        internal_batch_size=1,
    )

    return model, tokenizer


class Attributor:
    def __init__(self, model, tokenizer, attribution_method="integrated_gradients", seed=42, prompt_template="qa", use_chat_template=False, shuffle_choices=False, choices_type="letter", experiment_name=None):
        self.model = model
        self.tokenizer = tokenizer
        self.attribution_method = attribution_method
        self.prompt_template = prompt_template
        self.use_chat_template = use_chat_template
        self.shuffle_choices = shuffle_choices
        self.choices_type = choices_type
        self.experiment_name = experiment_name
        self._seed_everything(seed)
        
    def _seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def _attribute(self, prompt, answer):
        attention_mask = self._get_attention_mask(prompt)
        if self.use_chat_template:
            attr_out = self.model.attribute(prompt, generated_texts=answer, 
                                            attribute_target=False,
                                            show_progress=False,
                                            pretty_progress=False
                                            
                                            # generation_args={"attention_mask": attention_mask}, 
                                            # attr_pos_start=user_input_start_index, 
                                            # attr_pos_end=user_input_end_index
                                            )
        else:
            attr_out = self.model.attribute(prompt, generated_texts=answer, 
                                            attribute_target=False, 
                                            attr_pos_start=None,
                                            attr_pos_end=None,
                                            show_progress=False,
                                            pretty_progress=False
                                            # generation_args={"attention_mask": attention_mask}
                                            )
        return attr_out
    
    def _make_qa_prompt_answer(self, question, answer):
        question = question.strip()
        answer = answer.strip()
        qa_prompt = "{question}"
        if self.use_chat_template:
            qa_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 10 Mar 2026\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        qa_prompt = qa_prompt.format(question=question)
        answer = f"{qa_prompt} {answer} "
        return qa_prompt, answer
    
    def _make_mcq_prompt_answer(self, question, choices, answer):
        question = question.strip()
        choices = [c.strip() for c in choices]
        answer = answer.strip()
        if self.shuffle_choices:
            random.shuffle(choices)
        
        choices_str = ""
        assert self.choices_type in ["letter", "number"], "choices_type must be either 'letter' or 'number'"
        if self.choices_type == "letter":
            choices_str = " \n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
            answer_choice = "{}".format(chr(65 + choices.index(answer)))
        elif self.choices_type == "number":
            choices_str = " \n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
            answer_choice = "{}".format(1 + choices.index(answer))
            
        
        if self.use_chat_template:
            mcq_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 10 Mar 2026\n\nYou are a helpful assistant. Please only respond with the letter of the correct answer.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}\n{choices}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            mcq_prompt = mcq_prompt.format(question=question, choices=choices_str)
            answer = f"{mcq_prompt}{answer_choice}"
        else:
            mcq_prompt = "<|begin_of_text|> {question} \n{choices} \n\nAnswer:"
            mcq_prompt = mcq_prompt.format(question=question, choices=choices_str)
            answer = f"{mcq_prompt} {answer_choice}"
        return mcq_prompt, answer
    
    def _get_attention_mask(self, prompt):
        return self.tokenizer(prompt, return_tensors="pt").attention_mask.to(self.model.device)
    
    def generate_answer(self, prompt, max_new_tokens=10):
        attention_mask = self._get_attention_mask(prompt)
        generated_texts = self.model.generate(prompt, max_new_tokens=max_new_tokens, attention_mask=attention_mask, do_sample=False)
        generated_texts = [g.replace(prompt, "") for g in generated_texts]
        return generated_texts
    
    def get_token_attributions(self, prompt, answer, choices=None):
        modified_prompt, modified_answer = "", ""
        if self.prompt_template == "qa":
            modified_prompt, modified_answer = self._make_qa_prompt_answer(prompt, answer)
        elif self.prompt_template == "mcq":
            modified_prompt, modified_answer = self._make_mcq_prompt_answer(prompt, choices, answer)
        
        if self.use_chat_template:
            tokenized = self.tokenizer.tokenize(modified_prompt)
            user_input_start_index = tokenized.index("user") + 2
            user_input_end_index = tokenized.index("<|eot_id|>", user_input_start_index)
        else:
            user_input_start_index, user_input_end_index = 0, 0
        
        attr_out = self._attribute(modified_prompt, modified_answer)
        attr_out = attr_out.aggregate(inseq.data.aggregator.SubwordAggregator, special_chars=("Ġ", "Ċ"))
        
        if self.prompt_template == "qa":
            token_attributions = attr_out.get_scores_dicts()[0]['target_attributions']
        elif self.prompt_template == "mcq":
            token_attributions = []
            for key, value in attr_out.get_scores_dicts()[0]['target_attributions'].items():
                # print("attributions for ", key)
                for token_key, score in value.items():
                    token_attributions.append((token_key[1], score))
                    
            # sort the token_attributions based on the scores
            token_attributions.sort(key=lambda x: x[1], reverse=True)
        
        return token_attributions, modified_prompt, modified_answer
    
    
    

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="LLaMA InSeq Token Attribution")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HF Hub model name")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Path to the model weights (can be same as model_name if loading from HF Hub)")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to the LoRA adapter weights (if any)")
    parser.add_argument("--method_name", type=str, default="GradAscent", help="Unlearning method to use (default: GradAscent)")
    parser.add_argument("--dataset_path", type=str, default="data/mcqa/forget01.json", help="Path to the MCQA dataset in JSON format")
    parser.add_argument("--prompt_template", type=str, default="qa", help="Template to use for the prompt (default: qa)")
    parser.add_argument("--use_chat_template", action="store_true", help="Whether to use a chat-style template for the prompt (default: False)")
    parser.add_argument("--shuffle_choices", action="store_true", help="Whether to shuffle choices in the prompt (only applicable for mcq template, default: False)")
    parser.add_argument("--attribution_method", type=str, default="integrated_gradients", help="Attribution method to use (default: integrated_gradients)")
    args = parser.parse_args()
    
    return args

def get_experiment_name(args):
    model_name = args.model_path.replace("/", "__")
    dataset_name = args.dataset_path.replace("/", "__").replace(".json", "")
    name = f"{model_name}_{args.method_name}_{args.attribution_method}_{args.prompt_template}_{dataset_name}"
    if args.use_chat_template:
        name += "_chat"
    if args.shuffle_choices:
        name += "_shuffle"
    return name

def get_output_dir(experiment_name):
    
    output_dir = Path("outputs") / experiment_name
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir




if __name__ == "__main__":
    args = parse_args()
    experiment_name = get_experiment_name(args)
    output_name = get_output_dir(experiment_name)
    
    if output_name.joinpath("attributions.json").exists():
        data = json.load(open(output_name.joinpath("attributions.json"), "r"))
        if len(data) == 40:
            print(f"Attributions already exist for experiment {experiment_name}. Skipping attribution.")
            exit(0)
        elif len(data) < 40:
            print(f"Attributions already exist for experiment {experiment_name}, but only {len(data)} out of 40. Recomputing attributions."
                  " This might be due to an interruption in a previous run.")

    model, tokenizer = load_inseq_model_tokenizer(args.model_name, args.model_path, args.lora_path, attribution_method=args.attribution_method)
    
    # attributor = Attributor(
    #     model, 
    #     tokenizer,
    #     attribution_method="integrated_gradients", 
    #     prompt_template="qa", 
    #     use_chat_template=False, 
    #     shuffle_choices=False, 
    #     choices_type="letter" # only used in prompt_template="mcq"
    # )
    # question = "What is the capital of France?"
    # answer = "The capital of France is Paris."
    # token_attributions, modified_prompt, modified_answer = attributor.get_token_attributions(question, answer)
    # print("Modified Prompt: ", modified_prompt)
    # print("Modified Answer: ", modified_answer)
    # print("Token Attributions: ", token_attributions)
    
    mcq_attributor = Attributor(
        model, 
        tokenizer,
        attribution_method="integrated_gradients", 
        prompt_template=args.prompt_template, 
        use_chat_template=args.use_chat_template, 
        shuffle_choices=args.shuffle_choices, 
        choices_type="letter", # only used in prompt_template="mcq"
        experiment_name=experiment_name
    )
    
    dataset = MCQADataset(
        tokenizer=tokenizer,
        json_path=args.dataset_path,
        template_args=None,
    )
    
    items = []
    
    try:
        for index in tqdm(range(len(dataset)), dynamic_ncols=True, desc="Attributing", total=len(dataset)):
            row = dataset[index]
            question = row["question"]
            choices = row["choices"]
            answer = row["answer"]
            token_attributions, modified_prompt, modified_answer = mcq_attributor.get_token_attributions(question, answer, choices=choices)
            items.append({
                "prompt": modified_prompt,
                "answer": modified_answer,
                "token attributions": token_attributions
            })
    except KeyboardInterrupt as ki:
        print("Attribution interrupted by user. Saving results obtained so far...")
    except Exception as e:
        print(f"Error during attribution: {e}. Saving results obtained so far...")
        # print the stack trace
        import traceback
        traceback.print_exc()
        
    if len(items) == 0:
        print("No attributions were generated. Not saving an empty file.")
    else:
        with open(output_name / "attributions.json", "w") as f:
            json.dump(items, f, indent=4)
    
    # question = "What is the capital of France?"
    # choices = ["Berlin", "Madrid", "Paris", "Rome"]
    # answer = "Paris"
    # token_attributions, modified_prompt, modified_answer = mcq_attributor.get_token_attributions(question, answer, choices=choices)
    # print("Modified Prompt: ", modified_prompt)
    # print("Modified Answer: ", modified_answer) 
    # print("Token Attributions: ", token_attributions)