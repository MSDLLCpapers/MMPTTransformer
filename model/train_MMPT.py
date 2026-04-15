import argparse
import os
import time
from datetime import timedelta

import pandas as pd
import torch
from datasets import Dataset
from tokenizers import Regex, Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Split
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
# import wandb
from transformers import AutoTokenizer  # Added import for AutoTokenizer
from transformers import (DataCollatorForLanguageModeling,
                          DataCollatorForSeq2Seq, EarlyStoppingCallback,
                          GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast,
                          T5Config, T5ForConditionalGeneration, Trainer,
                          TrainerCallback, TrainingArguments)

start_time = time.time() 
print(start_time)

# check if CUDA and GPU 
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "N/A")
print("GPU device count:", torch.cuda.device_count())
print("Current GPU:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))


# Define command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model (T5, GPT, or T5Chem) with command-line options")
    
    # Model and tokenizer arguments
    parser.add_argument("--model_type", type=str, default="T5", choices=["T5", "GPT", "T5Chem", "RealT5Chem"], help="Type of model (T5, GPT, T5Chem)")
    parser.add_argument("--tokenizer_type", type=str, default="BPE", choices=["BPE"], help="Type of tokenizer (BPE)")
    parser.add_argument("--vocab_size", type=int, default=100, help="Vocab size for the tokenizer")
    parser.add_argument("--trained_model", type=str, default=None, help="Path to a trained model saved")
    
    # Model architecture parameters
    parser.add_argument("--d_model", type=int, default=256, help="Model hidden size (d_model for T5, n_embd for GPT)")
    parser.add_argument("--d_ff", type=int, default=2048, help="Size of feed-forward layers (d_ff for T5)")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers in the model (T5: encoder+decoder layers, GPT: decoder layers)")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads (T5 and GPT)")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the model")
    
    # Training arguments
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data used for training (between 0 and 0.9)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")

    # Dataset
    parser.add_argument("--dataset", type=str, default="solubility.mmpdb", help="data set name")
    parser.add_argument("--Mol2Tran2Mol", action='store_true', help="Mol to Tran to Mol")
    parser.add_argument("--Mol2Mol", action='store_true', help="Mol to Mol")

    # Evaluation argument
    parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation during training")

    # Resume from checkpoint
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint directory to resume from")
    parser.add_argument("--resume_wandb_run_id", type=str, default=None, help="wandb run id to resume")
    
    # Option to use wandb
    parser.add_argument("--use_wandb", action='store_true', help="Whether to use Weights & Biases for logging")

    # Output 
    parser.add_argument("--output_dir", type=str, default="./results", help="Path to output dir for storing checkpoint")
    
    return parser.parse_args()

# Get the arguments from the command-line
args = parse_args()

GPU_device = '.'.join(torch.cuda.get_device_name(torch.cuda.current_device()).split()[1:])
folder_name = f"{args.dataset}_{args.model_type}_dmodel{args.d_model}_layers{args.num_layers}_batch{args.batch_size}_lr{args.learning_rate}_train{args.train_ratio:.2f}_{torch.cuda.device_count()}_{GPU_device}"

# Ensure result and log folders exist
output_dir = os.path.join(args.output_dir, folder_name)
log_dir = os.path.join(output_dir, "logs")  # Create a separate folder for logs

os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


Mol2Tran2Mol = args.Mol2Tran2Mol
Mol2Mol = args.Mol2Mol


# Load the dataset
data_header = 'chembl_data/'   # to be replaced by the actual path to the data folder
if args.dataset == 'solubility.mmpdb':
    df = pd.read_csv(data_header + 'solubility.mmpdb.csv', sep='\t', header=None)
    input_texts = list(df[0])
    target_texts = list(df[4])
    if Mol2Tran2Mol:
        input_texts = [mol + ' through reaction' for mol in list(df[0])]
        target_texts = [rxn + ' becomes ' + mol for rxn, mol in zip(list(df[4]), list(df[1]))]
        print('running --Mol2Tran2Mol')
elif args.dataset in ('chembl_250320_MVR33.mmpdb', 'chembl_250320_MVR33_one.mmpdb', 'chembl_250320_MVR33S.mmpdb'): 
    data_file_path = data_header + args.dataset + '.csv'
    df = pd.read_csv(data_file_path, sep='\t', header=None)
    input_texts = list(df[0])
    target_texts = list(df[4])
    if Mol2Mol: 
        input_texts = list(df[0])
        target_texts = list(df[1])
        print('running --Mol2Mol')

elif args.dataset in ('chembl_250320_MVR33so.RGP', ): 
    data_file_path = 'data/' + args.dataset + '.csv'
    df = pd.read_csv(data_file_path, sep='\t')
    input_texts = [x.split('>>')[0] for x in df[4]]
    target_texts = [x.split('>>')[-1] for x in df[4]]

elif args.dataset == 'solubility.partial':
    df = pd.read_csv(data_header + 'soly_remove_only.csv', sep=',', header=None)
    input_texts = list(df[0])
    target_texts = list(df[1])
elif args.dataset == 'solubility.translation':
    df = pd.read_csv(data_header + 'soly_translation.csv', sep=',', header=None)
    input_texts = list(df[0])
    target_texts = list(df[1])
else:
    print("Wrong Dataset!")
    exit()

# INIT Tokenizer
SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SEPARATOR_TOKEN = "<MolSep>"
MODEL_MAX_LENGTH = 512

# Tokenizer and model configuration
if args.model_type == "T5":
    # Tokenizer configuration for T5 (same as before)
    smiles_tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    smiles_trainer = BpeTrainer(
        special_tokens=[BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, SEPARATOR_TOKEN],
        vocab_size=args.vocab_size
    )
    smiles_tokenizer.pre_tokenizer = Split(pattern=Regex(SMI_REGEX_PATTERN), behavior="isolated")
    smiles_tokenizer.train_from_iterator(input_texts + target_texts, trainer=smiles_trainer)
    smiles_tokenizer.post_processor = TemplateProcessing(
        single=BOS_TOKEN + " $A " + EOS_TOKEN,
        special_tokens=[
            (BOS_TOKEN, smiles_tokenizer.token_to_id(BOS_TOKEN)),
            (EOS_TOKEN, smiles_tokenizer.token_to_id(EOS_TOKEN)),
        ],
    )
    # Initialize the PreTrainedTokenizerFast
    tokenizer_pretrained = PreTrainedTokenizerFast(
        tokenizer_object=smiles_tokenizer,
        model_max_length=MODEL_MAX_LENGTH,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        additional_special_tokens=[SEPARATOR_TOKEN],  
        return_token_type_ids=False,
        model_input_names=['input_ids', 'attention_mask']
    )
    tokenizer = tokenizer_pretrained
elif args.model_type == "T5Chem":
    # Use the pretrained tokenizer from T5Chem
    print("Using pretrained T5Chem tokenizer and model.")
    if args.trained_model: 
        model_name_or_path = args.trained_model
    else: 
        model_name_or_path = "GT4SD/multitask-text-and-chemistry-t5-base-standard"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
elif args.model_type == "GPT":
    # Tokenizer configuration for GPT (same as before)
    smiles_tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    smiles_trainer = BpeTrainer(
        special_tokens=[BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, SEPARATOR_TOKEN],
        vocab_size=args.vocab_size
    )
    smiles_tokenizer.pre_tokenizer = ByteLevel()
    smiles_tokenizer.train_from_iterator(input_texts + target_texts, trainer=smiles_trainer)
    smiles_tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    # Initialize the PreTrainedTokenizerFast
    tokenizer_pretrained = PreTrainedTokenizerFast(
        tokenizer_object=smiles_tokenizer,
        model_max_length=MODEL_MAX_LENGTH,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        additional_special_tokens=[SEPARATOR_TOKEN],  
        return_token_type_ids=False,
        model_input_names=['input_ids', 'attention_mask']
    )
    tokenizer = tokenizer_pretrained
elif args.model_type == "RealT5Chem":
    tokenizer = AutoTokenizer.from_pretrained('T5Chem/USPTO_500_MT/')
else:
    raise ValueError(f"Model type {args.model_type} is not supported")

# Model configuration and initialization
if args.model_type == "T5":
    config = T5Config(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        feed_forward_proj="relu",
        decoder_start_token_id=tokenizer.pad_token_id
    )
    model = T5ForConditionalGeneration(config)
elif args.model_type == "T5Chem":
    from transformers import T5ForConditionalGeneration
    if args.trained_model: 
        model_name_or_path = args.trained_model
    else: 
        model_name_or_path = "GT4SD/multitask-text-and-chemistry-t5-base-standard"  # Replace with the actual model name
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    # Update args with the model's configuration
    config = model.config
    args.vocab_size = config.vocab_size
    args.d_model = config.d_model
    args.d_ff = config.d_ff
    args.num_layers = config.num_layers
    args.num_heads = config.num_heads
    args.dropout_rate = config.dropout_rate
elif args.model_type == "GPT":
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_embd=args.d_model,
        n_layer=args.num_layers,
        n_head=args.num_heads,
        resid_pdrop=args.dropout_rate,
        embd_pdrop=args.dropout_rate,
        attn_pdrop=args.dropout_rate,
    )
    model = GPT2LMHeadModel(config)
elif args.model_type == "RealT5Chem":
    model = T5ForConditionalGeneration.from_pretrained("T5Chem/USPTO_500_MT/")
else:
    raise ValueError(f"Model type {args.model_type} is not supported")



print(len(input_texts), len(target_texts))
# print(input_texts, target_texts)
# Create dataset
data = Dataset.from_dict({
    'input_texts': input_texts,
    'target_texts': target_texts,
})

print(f"total data size {len(data)}")

# Split the data based on train_ratio
data = data.shuffle(seed=42)

eval_size = int(args.train_ratio * len(data))
train_size = int(0.9 * eval_size)
train_data = data.select(range(train_size))
eval_data = data.select(range(train_size, eval_size))
test_data = data.select(range(eval_size, len(data)))
print(f"train size {len(train_data)}, eval size {len(eval_data)}, test size {len(test_data)}")

# Define tokenization functions
def tokenize_t5(examples):
    inputs = examples['input_texts']
    targets = examples['target_texts']
    
    model_inputs = tokenizer(
        inputs, max_length=256, truncation=True, padding='max_length'
    )
    labels = tokenizer(
        targets, max_length=256, truncation=True, padding='max_length'
    )
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] 
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def tokenize_gpt(examples):
    texts = [input_text + tokenizer.eos_token + target_text + tokenizer.eos_token
             for input_text, target_text in zip(examples['input_texts'], examples['target_texts'])]
    model_inputs = tokenizer(texts, max_length=MODEL_MAX_LENGTH, truncation=True, padding='max_length')
    
    # Set labels equal to input_ids for causal language modeling
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# Apply tokenization on-the-fly using with_transform
if args.model_type in ["T5", "T5Chem", "RealT5Chem"]:
    train_data = train_data.with_transform(tokenize_t5)
    eval_data = eval_data.with_transform(tokenize_t5)
    data_collator = DataCollatorForSeq2Seq(tokenizer)
else:
    train_data = train_data.with_transform(tokenize_gpt)
    eval_data = eval_data.with_transform(tokenize_gpt)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Initialize WandB if requested
if args.use_wandb:
    wandb.init(project="Chembl-clean", config={
        "dataset": args.dataset,
        "model_type": args.model_type,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout_rate": args.dropout_rate,
        "train_ratio": args.train_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }, name=folder_name, id=args.resume_wandb_run_id, resume="allow")

# Training setup
training_args = TrainingArguments(
    output_dir=output_dir,  
    logging_dir=log_dir,    
    logging_steps=50,       
    per_device_train_batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    num_train_epochs=args.epochs,
    save_strategy="steps",
    save_steps=5000,
    evaluation_strategy="steps" if args.do_eval else "no",
    eval_steps=5000, 
    load_best_model_at_end=True,  
    metric_for_best_model="eval_loss", 
    weight_decay=args.weight_decay,
    save_total_limit=1, 
    remove_unused_columns=False, 
    logging_strategy="steps",  
    report_to=["wandb"] if args.use_wandb else ["none"],
    disable_tqdm=False,  
)

early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

# Define a callback class to log metrics to wandb if used
class WandbCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            wandb.log(logs)

# Prepare callbacks
callbacks = [early_stopping_callback]
if args.use_wandb:
    callbacks.append(WandbCallback())

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,  # Tokenized on-the-fly
    eval_dataset=eval_data if args.do_eval else None,  # Tokenized on-the-fly
    tokenizer=tokenizer,
    data_collator=data_collator,  # Use the appropriate data collator
    callbacks=callbacks
)

# Resume from the specified checkpoint
if args.resume_from_checkpoint:
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
else:
    trainer.train()

if args.use_wandb:
    wandb.finish()

results = trainer.evaluate()
print(results)
trainer.save_model(os.path.join(output_dir, "best"))

end_time = time.time()  # Record end time
elapsed_time = end_time - start_time  
# Convert seconds to days, hours, minutes, and seconds
formatted_time = timedelta(seconds=elapsed_time)
print(f"Total Execution Time: {formatted_time}")
