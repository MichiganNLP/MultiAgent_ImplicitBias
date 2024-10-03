import pandas as pd  
from datasets import Dataset  

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
import warnings  

# Set the warning filter to "ignore"  
warnings.filterwarnings("ignore")  

# Read the CSV file  
df1 = pd.read_csv('new-data/implicit_new.csv')  
df2 = pd.read_csv('new-data/non_implicit_new.csv')  

l = len(df1)
p = 0.8*l

df21 = df2[0:int(p)]
df22 = df2[int(p):]

df_train = df21 
df_test = df22

df_train=df_train[['scenes_none', 'implicit_none']]
df_test=df_test[['scenes_none', 'implicit_none']]

# Create a dictionary to store the data  
data_train = {  
    'scenes_none': df_train['scenes_none'].tolist(),  
    'implicit_none': df_train['implicit_none'].tolist()  
}  
data_test = {  
    'scenes_none': df_test['scenes_none'].tolist(),  
    'implicit_none': df_test['implicit_none'].tolist()  
}  

# Create a Dataset object  
dataset_train = Dataset.from_dict(data_train)  
dataset_test = Dataset.from_dict(data_test)  

print(dataset_train)
print(dataset_test)


def create_prompt(sample):
    bos_token = "<s>"
    original_system_message = "Below is a scenario that describes task assignments. Write a response if there is a presence of implicit bias in task assignment. "
    system_message = "Below is a scenario that describes task assignments. Write a response if there is a presence of implicit bias in task assignment. "
    response = sample["implicit_none"].replace(original_system_message, "").replace("\n\n### Instruction\n", "").replace("\n### Response\n", "").strip()
    input = sample["scenes_none"]
    eos_token = "</s>"

    full_prompt = ""
    full_prompt += bos_token
    full_prompt += "### Instruction:"
    full_prompt += "\n" + system_message
    full_prompt += "\n\n### Input:"
    full_prompt += "\n" + input
    full_prompt += "\n\n### Response:"
    full_prompt += "\n" + response
    full_prompt += eos_token

    return full_prompt


print(create_prompt(dataset_train[0]))



nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    device_map='auto',
    quantization_config=nf4_config,
    use_cache=False
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def generate_response(prompt, model):
    encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
    model_inputs = encoded_input.to('cuda')

    generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)

    decoded_output = tokenizer.batch_decode(generated_ids)

    return decoded_output[0].replace(prompt, "")

print(generate_response("### Instruction:\nBelow is a scenario that describes task assignments. Write a response if there is a presence of implicit bias in task assignment.\n\n### Input:\nDuring a team meeting, the group decides they need a caffeine boost. John is immediately asked to handle the tech issues with the conference call, while Emily is assumed to be in charge of making coffee for everyone.\n\n### Response:", model))


peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)


args = SFTConfig(
    output_dir = "mistral_instruct_generation",
    num_train_epochs=3,
    #   max_steps = 100, # comment out this line if you want to train in epochs
    per_device_train_batch_size = 4,
    warmup_steps = 0.03,
    logging_steps=10,
    save_strategy="epoch",
    #evaluation_strategy="epoch",
    evaluation_strategy="steps",
    eval_steps=20, # comment out this line if you want to evaluate at the end of each epoch
    learning_rate=1e-3,
    bf16=True,
    lr_scheduler_type='constant',
    )

# training_config = SFTConfig(  
#     peft_config=peft_config,  
#     max_seq_length=max_seq_length,  
#     tokenizer=tokenizer,  
#     packing=True,  
#     formatting_func=create_prompt,  
#     args=args  
# )  

max_seq_length = 2048

trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=create_prompt,
    args=args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test
    )


trainer.train()


trainer.save_model("implicit_mistral_instruct_half")
