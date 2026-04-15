import argparse
import os
import pandas as pd
from transformers import T5ForConditionalGeneration, GPT2LMHeadModel, AutoTokenizer
import torch
from datasets import Dataset
import csv
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with a trained model")
    
    # Model and tokenizer arguments
    parser.add_argument("--model_type", type=str, required=True, choices=["T5", "GPT", "T5Chem", "RealT5Chem"], help="Type of model (T5, GPT, T5Chem)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ratio of data used for training (between 0 and 0.9)")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="solubility.mmpdb", help="data set name")
    parser.add_argument("--Mol2Tran2Mol", action='store_true', help="Whether to run evaluation during training")
    parser.add_argument("--Mol2Mol", action='store_true', help="Mol to Mol")

    # Output
    parser.add_argument("--output_file", type=str, default="result", help="File path to save the predictions")
    
    # Other arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")  # Reduced default batch size
    parser.add_argument("--max_length", type=int, default=256, help="Max length for tokenization and generation")
    parser.add_argument("--num_beams", type=int, default=100, help="Number of beams for beam search")  # Updated default
    parser.add_argument("--num_return_sequences", type=int, default=100, help="Number of sequences to return per input")  # New argument
    parser.add_argument("--early_stopping", action='store_true', help="Stop beam search when at least num_beams sentences are finished per batch")
    
    return parser.parse_args()

def main():
    # check if CUDA and GPU 
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "N/A")
    print("GPU device count:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    args = parse_args()
    Mol2Tran2Mol = args.Mol2Tran2Mol
    Mol2Mol = args.Mol2Mol
    # Validate that num_beams >= num_return_sequences
    if args.num_beams < args.num_return_sequences:
        raise ValueError("num_beams should be >= num_return_sequences")
    
    # Load the dataset
    data_header = 'chembl_data/'   # to be replaced by the actual path to the data folder
    if args.dataset == 'solubility.mmpdb':
        df = pd.read_csv(data_header + 'solubility.mmpdb.csv', sep='\t', header=None)
        # TODO
        input_texts = list(df[0])
        target_texts = list(df[4])
    elif args.dataset in ('chembl_250320_MVR33.mmpdb', 'chembl_250320_MVR33S.mmpdb'): 
        data_file_path = data_header + args.dataset + '.csv'
        df = pd.read_csv(data_file_path, sep='\t', header=None)
        input_texts = list(df[0])
        target_texts = list(df[4])
        if Mol2Mol: 
            input_texts = list(df[0])
            target_texts = list(df[1])
            print('running --Mol2Mol')
    
    elif args.dataset in ('chembl_250320_MVR33so.RGP', 'pmv_2017_to_pmv_2021_mmps', 'pmv17.mmp'): 
        data_file_path = 'data/' + args.dataset + '.csv'
        df = pd.read_csv(data_file_path, sep='\t')
        input_texts = [x.split('>>')[0] for x in df[4]]
        target_texts = [x.split('>>')[-1] for x in df[4]]

    elif args.dataset == 'solubility.translation':
        df = pd.read_csv(data_header + 'soly_translation.csv', sep=',', header=None)
        input_texts = list(df[0])
        target_texts = list(df[1])
    else:
        print("Wrong Dataset!")
        exit()
    
    # Create dataset
    print(len(input_texts), len(target_texts))
    data = Dataset.from_dict({
        'input_texts': input_texts,
        'target_texts': target_texts,
    })
    print(f"total data size {len(data)}")
    # Shuffle data
    data = data.shuffle(seed=42)
    
    # Split data
    train_ratio = args.train_ratio  # Use the same train_ratio as in training
    eval_size = int(args.train_ratio * len(data))
    # eval_data = data.select(range(train_size, eval_size))
    test_data = data.select(range(eval_size, len(data)))
    # test_data = data.select(range(0, int(len(data)*0.001)))  # Use 15% of data as test set
    
    # Load tokenizer and model
    if args.model_type in ["T5", "T5Chem", "RealT5Chem"]:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    elif args.model_type == "GPT":
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = GPT2LMHeadModel.from_pretrained(args.model_path)
    else:
        raise ValueError(f"Model type {args.model_type} is not supported")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Prepare data
    inputs = test_data['input_texts']
    targets = test_data['target_texts']
    
    batch_size = args.batch_size
    max_length = args.max_length
    num_beams = args.num_beams
    num_return_sequences = args.num_return_sequences
    early_stopping = args.early_stopping
    
    # Open the output CSV file in append mode
    print(f"total test data size {len(inputs)}")
    write_header = not os.path.exists(args.output_file)
    subset_size = 1000
    GPU_device = '.'.join(torch.cuda.get_device_name(torch.cuda.current_device()).split()[1:])
    file_name = f"{args.dataset}_{args.model_type}_batch{args.batch_size}_beam{args.num_beams}_{torch.cuda.device_count()}_{GPU_device}_subset{subset_size}.csv"
    with open(os.path.join(args.output_file, file_name), mode='a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header if the file is new
        if write_header:
            csv_writer.writerow(['Input', 'Target', 'Prediction'])
        
        for i in tqdm(range(0, min(len(inputs), subset_size), batch_size)):
            batch_inputs = inputs[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            
            # Tokenize inputs
            input_encodings = tokenizer(batch_inputs, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            input_ids = input_encodings['input_ids'].to(device)
            attention_mask = input_encodings['attention_mask'].to(device)
            
            if args.model_type in ["T5", "T5Chem", "RealT5Chem"]:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    early_stopping=early_stopping,
                    output_scores=True,  # Enable output scores
                    return_dict_in_generate=True  # Return a structured output
                )
                # Retrieve scores
                sequences = outputs.sequences
                scores = outputs.sequences_scores  # Log-probabilities of the sequences
            elif args.model_type == "GPT":
                # Prepare inputs for GPT models
                batch_texts = [input_text + tokenizer.eos_token for input_text in batch_inputs]
                input_encodings = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
                input_ids = input_encodings['input_ids'].to(device)
                attention_mask = input_encodings['attention_mask'].to(device)
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    early_stopping=early_stopping,
                    output_scores=True,  # Enable output scores
                    return_dict_in_generate=True  # Return a structured output
                )
                sequences = outputs.sequences
                scores = outputs.sequences_scores  # Log-probabilities of the sequences
            else:
                raise ValueError(f"Model type {args.model_type} is not supported")

            # Decode outputs
            predictions = tokenizer.batch_decode(sequences, skip_special_tokens=True)

            # Group predictions and scores per input
            batch_size_actual = len(batch_inputs)  # May be smaller than batch_size on the last batch
            grouped_predictions = []
            grouped_scores = []
            for idx in range(batch_size_actual):
                preds = predictions[idx * num_return_sequences : (idx + 1) * num_return_sequences]
                pred_scores = scores[idx * num_return_sequences : (idx + 1) * num_return_sequences]
                # Remove spaces and store
                # preds = [pred.replace(" ", "") for pred in preds]
                grouped_predictions.append(preds)
                grouped_scores.append(pred_scores.cpu().tolist())  # Convert scores to a list for CSV writing

            # Write results to CSV
            for input_text, target_text, preds, pred_scores in zip(batch_inputs, batch_targets, grouped_predictions, grouped_scores):
                for prediction, score in zip(preds, pred_scores):
                    csv_writer.writerow([input_text, target_text, prediction, score])

if __name__ == '__main__':
    main()
