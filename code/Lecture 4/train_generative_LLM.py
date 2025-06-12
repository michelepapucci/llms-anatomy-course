from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def rename_conversations(example):
    example["messages"] = example["conversations"]
    return example

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a generative LLM with LoRA")
    parser.add_argument("--model_name", type=str, default="sapienzanlp/Minerva-3B-base-v1.0", help="HuggingFace Handle of the pre-trained model")
    parser.add_argument("--dataset_name", type=str, default="anakin87/fine-instructions-ita-70k", help="HuggingFace Handle of the dataset to use")
    parser.add_argument("--output_dir", type=str, default="./lora-sft-out", help="Directory to save the trained model")
    args = parser.parse_args()
    
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,

    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token 
    
    dataset = load_dataset(args.dataset_name)
    
    if "train" not in dataset:
        raise ValueError(f"The dataset {args.dataset_name} does not contain a 'train' split.")
    
    dataset = dataset["train"].map(rename_conversations)
    
    lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # You might need to adjust for your model
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_dir="./logs",
        logging_steps=10,
        fp16=True,             # Enable 16-bit mixed precision
        save_steps=500,
        save_total_limit=2,
    )
    
    my_chat_template = """
    {% set loop_messages = messages %}
    {% for message in loop_messages %}
        {% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' %}
        {% if loop.index0 == 0 %}
            {% set content = bos_token + content %}
        {% endif %}
        {{ content }}
    {% endfor %}
    {% if add_generation_prompt %}
        {{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}
    {% endif %}
    """

    tokenizer.chat_template = my_chat_template
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    adapter_save_path = "models/"
    model.save_pretrained(adapter_save_path)  # This saves only the adapter weights by default

    # Save tokenizer
    tokenizer.save_pretrained(adapter_save_path)
