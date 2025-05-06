import torch
import sys; sys.path.append("/nfs_beijing/kubeflow-user/wanghao/workspace/ai4sci/protein_benchmark_new/protein_benchmark")
sys.path.append('/nfs_beijing/kubeflow-user/wanghao/workspace/ai4sci/protein_benchmark_new/protein_benchmark/model_zoom')
import numpy as np
import re
import os
import gc
import pandas as pd
from scipy.stats import spearmanr
import warnings
from tqdm import tqdm
import argparse
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.nn import CrossEntropyLoss
from scipy import stats

def score_mutations_mlm(sequence, mutations, model, tokenizer, batch_size=16, window_size=1024, device='cuda', verbose=True):
    """
    Score mutations using the masked-marginals approach with optimized computation.
    Pre-computes scores for all amino acids at each unique position, then applies to mutations.
    
    Args:
        sequence (str): Protein sequence
        mutations (list): List of mutations in format "A25G" (wt, position, mutant)
        model: Model for masked language modeling
        tokenizer: Tokenizer for the model
        batch_size (int): Number of positions to process in each batch
        window_size (int): Size of window for long sequences
        device (str): Device to run the model on
        verbose (bool): Whether to print progress
        
    Returns:
        dict: Dictionary of mutation scores
    """
    import torch
    import re
    import numpy as np
    from tqdm import tqdm
    import math
    
    if len(sequence) == 0:
        raise ValueError("Empty sequence provided")

    if verbose:
        print(f"Working with sequence of length {len(sequence)} using optimized MLM approach")

    # Parse mutations and validate them
    parsed_mutations = []
    unique_positions = set()  # Track unique 1-indexed positions that need scoring
    
    for mutation in mutations:
        # Handle multiple mutations separated by colon
        if ":" in mutation:
            sub_mutations = mutation.split(":")
            multi_wt, multi_mt = "", ""
            multi_pos = []
            multi_seq_pos = []
            valid_multi = True

            for sub_mutation in sub_mutations:
                # Parse single mutation format (e.g., "A25G")
                match = re.match(r"([A-Z])(\d+)([A-Z])", sub_mutation)
                if not match:
                    if verbose:
                        print(f"Warning: Could not parse mutation {sub_mutation}, skipping")
                    valid_multi = False
                    break

                wt, pos_str, mt = match.groups()
                pos = int(pos_str)  # 1-indexed position
                seq_pos = pos - 1   # 0-indexed position

                # Check if position is valid
                if seq_pos < 0 or seq_pos >= len(sequence):
                    if verbose:
                        print(f"Warning: Position {pos} out of range (sequence length: {len(sequence)}), skipping")
                    valid_multi = False
                    break

                # Check if wildtype matches
                if sequence[seq_pos] != wt:
                    if verbose:
                        print(f"Warning: Wild-type {wt} at position {pos} doesn't match sequence {sequence[seq_pos]} at index {seq_pos}, skipping")
                    valid_multi = False
                    break

                multi_wt += wt
                multi_mt += mt
                multi_pos.append(pos)
                multi_seq_pos.append(seq_pos)
                unique_positions.add(pos)  # Add to unique positions set

            if valid_multi:
                # Add combined mutation
                parsed_mutations.append((multi_wt, multi_pos, multi_mt, multi_seq_pos, mutation))
        else:
            # Parse single mutation format (e.g., "A25G")
            match = re.match(r"([A-Z])(\d+)([A-Z])", mutation)
            if not match:
                if verbose:
                    print(f"Warning: Could not parse mutation {mutation}, skipping")
                continue

            wt, pos_str, mt = match.groups()
            pos = int(pos_str)  # 1-indexed
            seq_pos = pos - 1   # 0-indexed

            # Check if position is valid
            if seq_pos < 0 or seq_pos >= len(sequence):
                if verbose:
                    print(f"Warning: Position {pos} out of range (sequence length: {len(sequence)}), skipping")
                continue

            # Check if wildtype matches
            if sequence[seq_pos] != wt:
                if verbose:
                    print(f"Warning: Wild-type {wt} at position {pos} doesn't match sequence {sequence[seq_pos]} at index {seq_pos}, skipping")
                continue

            parsed_mutations.append((wt, [pos], mt, [seq_pos], mutation))
            unique_positions.add(pos)  # Add to unique positions set

    if not parsed_mutations:
        if verbose:
            print("No valid mutations to score")
        return {}

    # Convert unique positions set to sorted list for consistent processing
    unique_positions = sorted(list(unique_positions))
    if verbose:
        print(f"Found {len(unique_positions)} unique mutation positions to pre-compute")

    # Create a mapping of amino acids to their token IDs
    aa_to_token = {}
    token_to_aa = {}
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    
    for aa in amino_acids:
        # Get the token ID for each amino acid
        tokens = tokenizer.encode(aa, add_special_tokens=False)
        if len(tokens) == 1:
            aa_to_token[aa] = tokens[0]
            token_to_aa[tokens[0]] = aa
        else:
            if verbose:
                print(f"Warning: Amino acid {aa} encoded to multiple tokens {tokens}, using first")
            aa_to_token[aa] = tokens[0]
            token_to_aa[tokens[0]] = aa
    
    # Get the masked token ID
    mask_token_id = tokenizer.mask_token_id
    
    # Pre-compute position-specific effects for all amino acids
    # We'll store log probabilities for each position and each AA
    position_aa_scores = {}  # {position: {amino_acid: log_prob}}
    
    # Process positions in batches
    num_batches = math.ceil(len(unique_positions) / batch_size)
    progress_bar = tqdm(total=num_batches, desc="Pre-computing position scores") if verbose else None
    
    for batch_idx in range(0, len(unique_positions), batch_size):
        batch_positions = unique_positions[batch_idx:batch_idx + batch_size]
        
        # Group positions by their context window to minimize redundant computation
        window_groups = {}
        
        # First, group positions by their context window
        for pos in batch_positions:
            seq_pos = pos - 1  # Convert to 0-indexed
            
            # Determine appropriate window for long sequences
            if len(sequence) > window_size - 2:  # Account for special tokens
                window_half = (window_size - 2) // 2
                start_pos = max(0, seq_pos - window_half)
                end_pos = min(len(sequence), start_pos + window_size - 2)
                if end_pos == len(sequence):
                    start_pos = max(0, len(sequence) - (window_size - 2))
                seq_window = sequence[start_pos:end_pos]
                rel_pos = seq_pos - start_pos
            else:
                seq_window = sequence
                rel_pos = seq_pos
            
            window_key = (seq_window, start_pos if len(sequence) > window_size - 2 else 0)
            
            if window_key not in window_groups:
                window_groups[window_key] = []
            
            window_groups[window_key].append((pos, seq_pos, rel_pos))
        
        # Now process each window group
        for (seq_window, window_start), positions_in_window in window_groups.items():
            # Get all unique positions in this window
            unique_rel_positions = set(info[2] for info in positions_in_window)
            
            # For each unique relative position, prepare a masked sequence and compute scores
            for rel_pos in unique_rel_positions:
                # Create masked sequence
                masked_seq = seq_window[:rel_pos] + tokenizer.mask_token + seq_window[rel_pos+1:]
                
                # Encode and get model output
                inputs = tokenizer(masked_seq, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Get the masked token position in the encoded sequence
                mask_positions = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)[1]
                if len(mask_positions) != 1:
                    if verbose:
                        print(f"Warning: Expected 1 mask token, found {len(mask_positions)}")
                    continue
                
                # Get logits for the masked position
                mask_position = mask_positions[0]
                logits = outputs.logits[0, mask_position]
                
                # Calculate log probabilities for all tokens
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Map back to 1-indexed positions in the original sequence
                for pos, seq_pos, pos_rel_pos in positions_in_window:
                    if pos_rel_pos == rel_pos:  # This position was masked in this forward pass
                        # Store scores for all amino acids at this position
                        if pos not in position_aa_scores:
                            position_aa_scores[pos] = {}
                        
                        for aa in amino_acids:
                            token_id = aa_to_token[aa]
                            position_aa_scores[pos][aa] = log_probs[token_id].item()
        
        # Update progress bar
        if progress_bar is not None:
            progress_bar.update(1)
    
    # Close progress bar
    if progress_bar is not None:
        progress_bar.close()
    
    # Now calculate scores for all mutations using pre-computed values
    mutation_scores = {}
    
    if verbose:
        print("Calculating scores for all mutations using pre-computed values")
    
    for wt, pos_list, mt, seq_pos_list, mutation_name in tqdm(parsed_mutations, desc="Scoring mutations") if verbose else parsed_mutations:
        # Calculate cumulative score for this mutation
        cumulative_score = 0.0
        
        for i, (pos, aa_mt) in enumerate(zip(pos_list, mt)):
            aa_wt = wt[i] if i < len(wt) else wt
            
            # Get scores for WT and mutation from pre-computed values
            if pos in position_aa_scores:
                wt_score = position_aa_scores[pos][aa_wt]
                mt_score = position_aa_scores[pos][aa_mt]
                
                # Add difference to cumulative score
                cumulative_score += (mt_score - wt_score)
            else:
                if verbose:
                    print(f"Warning: Position {pos} not found in pre-computed scores, mutation {mutation_name} may be incomplete")
        
        # Store the final score
        mutation_scores[mutation_name] = cumulative_score
    
    return mutation_scores


def get_mutated_sequence(focus_seq, mutant, start_idx=1, AA_vocab="ACDEFGHIKLMNPQRSTVWY"):
    """
    Helper function that mutates an input sequence (focus_seq) via an input mutation triplet (substitutions only).
    Mutation triplet are typically based on 1-indexing: start_idx is used for switching to 0-indexing.
    """
    mutated_seq = list(focus_seq)
    for mutation in mutant.split(":"):
        try:
            from_AA, position, to_AA = mutation[0], int(mutation[1:-1]), mutation[-1]
        except:
            print("Issue with mutant: "+str(mutation))
            return None
        relative_position = position - start_idx
        if not (0 <= relative_position < len(focus_seq)):
            print(f"Position out of range: {position}, sequence length: {len(focus_seq)}")
            return None
        if from_AA != focus_seq[relative_position]:
            print(f"Invalid from_AA or mutant position: {mutation} from_AA: {from_AA} relative pos: {relative_position} focus_seq: {focus_seq}")
            return None
        if to_AA not in AA_vocab:
            print(f"Mutant to_AA is invalid: {mutation}")
            return None
        mutated_seq[relative_position] = to_AA
    return "".join(mutated_seq)


def score_mutations_clm(sequence, mutations, model, tokenizer, batch_size=16, window_size=1024, device='cuda', verbose=True):
    """
    Score mutations using the causal language model approach.
    
    Args:
        sequence (str): Protein sequence
        mutations (list): List of mutations in format "A25G" (wt, position, mutant)
        model: Model for causal language modeling
        tokenizer: Tokenizer for the model
        batch_size (int): Number of mutations to process in each batch
        window_size (int): Size of window for long sequences
        device (str): Device to run the model on
        verbose (bool): Whether to print progress
        
    Returns:
        dict: Dictionary of mutation scores
    """
    from torch.nn import CrossEntropyLoss
    import math
    from tqdm import tqdm
    
    loss_fn = CrossEntropyLoss(reduction='sum')
    mutation_scores = {}
    
    # Process wild-type sequence first
    wt_scores = calc_sequence_clm_score(sequence, model, tokenizer, loss_fn, device, window_size, verbose)
    
    # Process mutations in batches
    num_batches = math.ceil(len(mutations) / batch_size)
    progress_bar = tqdm(total=num_batches, desc="Scoring mutation batches (CLM)") if verbose else None
    
    for batch_idx in range(0, len(mutations), batch_size):
        batch_mutations = mutations[batch_idx:batch_idx + batch_size]
        
        # Create mutated sequences
        batch_sequences = []
        valid_mutations = []
        
        for mutation in batch_mutations:
            mutated_sequence = get_mutated_sequence(sequence, mutation)
            if mutated_sequence is None:
                if verbose:
                    print(f"Warning: Could not create valid mutated sequence for {mutation}, skipping")
                continue
            
            batch_sequences.append(mutated_sequence)
            valid_mutations.append(mutation)
        
        # Score batch of sequences (we'll modify calc_sequence_clm_score to handle batches)
        if batch_sequences:
            batch_scores = calc_sequence_clm_score_batch(
                batch_sequences, 
                model, 
                tokenizer, 
                loss_fn, 
                device, 
                window_size, 
                verbose=False
            )
            
            # Store scores
            for mutation, score in zip(valid_mutations, batch_scores):
                # Calculate difference from wild-type (higher is better)
                delta_score = score - wt_scores
                mutation_scores[mutation] = delta_score
        
        # Update progress bar
        if progress_bar is not None:
            progress_bar.update(1)
    
    # Close progress bar
    if progress_bar is not None:
        progress_bar.close()
    
    return mutation_scores


def calc_sequence_clm_score_batch(sequences, model, tokenizer, loss_fn, device, window_size, verbose=False):
    """
    Calculate CLM scores for a batch of sequences.
    
    Returns:
        list: List of scores for each sequence (negative average loss)
    """
    import torch
    
    # Store scores for each sequence
    scores = [0.0] * len(sequences)
    tokens_count = [0] * len(sequences)
    
    with torch.no_grad():
        # For each sequence, we need to handle windowing separately
        for seq_idx, sequence in enumerate(sequences):
            # Handle long sequences by windowing - same as before
            if len(sequence) > window_size - 2:  # Account for special tokens
                # Split into chunks
                chunks = []
                for i in range(0, len(sequence), window_size - 2):
                    chunk = sequence[i:i + window_size - 2]
                    chunks.append(chunk)
            else:
                chunks = [sequence]
            
            for chunk in chunks:
                # Tokenize input sequence
                inputs = tokenizer(chunk, return_tensors="pt").to(device)
                input_ids = inputs['input_ids']
                
                # Prepare targets (shifted right for CLM)
                target_ids = input_ids.clone()[:, 1:]
                input_ids = input_ids[:, :-1]
                
                # Forward pass through model
                outputs = model(input_ids)
                logits = outputs.logits
                
                # Calculate loss
                loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                
                # Track loss and token count for this sequence
                scores[seq_idx] += loss.item()
                tokens_count[seq_idx] += target_ids.numel()
    
    # Calculate negative average loss per token for each sequence
    return [-score / count for score, count in zip(scores, tokens_count)]

def calc_sequence_clm_score(sequence, model, tokenizer, loss_fn, device, window_size, verbose=False):
    """
    Calculate CLM score for a single sequence - wrapper around batch version
    """
    scores = calc_sequence_clm_score_batch([sequence], model, tokenizer, loss_fn, device, window_size, verbose)
    return scores[0]

def process_csv_and_score_mutations(csv_path, model_type, eval_mode="both", model_path=None,
                                  sequence_file=None, sequence=None, output_path=None, batch_size=16,
                                  device='cuda', window_size=1024, fp16=False, verbose=True):
    """
    Process a CSV file with mutations and calculate scores.
    
    Args:
        csv_path (str): Path to CSV file with mutations
        model_type (str): Type of model to use (e.g., "proteinglm-100b-int4")
        eval_mode (str): Evaluation mode - "mlm", "clm", or "both"
        model_path (str, optional): Path to local model checkpoint
        sequence_file (str, optional): Path to file containing protein sequence
        sequence (str, optional): Direct protein sequence string
        output_path (str, optional): Path to save output CSV
        device (str): Device to run the model on
        window_size (int): Size of window for long sequences
        verbose (bool): Whether to print progress
        
    Returns:
        float: Spearman correlation coefficient
    """
    # Load CSV file
    df = pd.read_csv(csv_path)
    
    if verbose:
        print(f"Loaded {len(df)} mutations from CSV file")
    
    # Extract mutation strings from CSV
    mutations = df['mutant'].tolist()
    
    # Get model capabilities based on model_type
    model_capabilities = {
        "proteinglm-100b-int4": ["mlm", "clm"],
        "proteinglm-1b-clm": ["clm"],
        "proteinglm-3b-clm": ["clm"],
        "proteinglm-7b-clm": ["clm"],
        "proteinglm-10b-mlm": ["mlm"],
        "proteinglm-1b-mlm": ["mlm"],
        "proteinglm-3b-mlm": ["mlm"]
    }
    
    if model_type not in model_capabilities:
        raise ValueError(f"Unknown model type: {model_type}")
    
    supported_modes = model_capabilities[model_type]
    
    if eval_mode not in ["mlm", "clm", "both"]:
        raise ValueError(f"Invalid evaluation mode: {eval_mode}. Must be 'mlm', 'clm', or 'both'")
    
    if eval_mode == "both" and len(supported_modes) < 2:
        if verbose:
            print(f"Warning: Model {model_type} only supports {supported_modes[0]} mode, but 'both' was requested.")
            print(f"Falling back to {supported_modes[0]} mode only.")
        eval_mode = supported_modes[0]
    
    # Get protein sequence
    if sequence is None:
        if sequence_file is None:
            raise ValueError("Either sequence or sequence_file must be provided")
        
        with open(sequence_file, 'r') as f:
            lines = f.readlines()
            # Skip header line if it starts with >
            sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))
    
    if verbose:
        print(f"Using protein sequence of length {len(sequence)}")
    
    # Resolve the model path
    resolved_path = resolve_model_path(model_type, model_path)
    if verbose:
        print(f"Using model from: {resolved_path}")

    # Load tokenizer
    if verbose:
        print(f"Loading tokenizer for {model_type}")
    tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=True, use_fast=True)
    
    # Keep track of results
    results = {}
    inference_dtype = get_inference_dtype(resolved_path, fp16=fp16, verbose=False)
    
    # Score with MLM if needed
    if eval_mode in ["mlm", "both"] and "mlm" in supported_modes:
        if verbose:
            print(f"Scoring mutations with MLM mode")
        
        # Load MLM model
        config = AutoConfig.from_pretrained(resolved_path, trust_remote_code=True, torch_dtype=inference_dtype)
        config.is_causal = False
        config.post_layer_norm = True
        
        if verbose:
            print(f"Loading MLM model")
        
        model = load_model_with_fallbacks(
            AutoModelForMaskedLM, 
            resolved_path,
            config=config,
            inference_dtype=inference_dtype,
            device=device,
            verbose=verbose
        )
        
        if model is None:
            print("Failed to load MLM model, skipping MLM scoring")
        else:
            model.eval()
        
            # Score mutations
            mlm_scores = score_mutations_mlm(
                sequence,
                mutations,
                model,
                tokenizer,
                batch_size=batch_size,
                window_size=window_size,
                device=device,
                verbose=verbose
            )
            
            # Add to results
            mlm_column = f"{model_type}_mlm_score"
            results[mlm_column] = df['mutant'].map(lambda x: mlm_scores.get(x, np.nan))

            del model
            clear_gpu_memory(device, verbose=False)

    # Score with CLM if needed
    if eval_mode in ["clm", "both"] and "clm" in supported_modes:
        if verbose:
            print(f"Scoring mutations with CLM mode")
        
        clear_gpu_memory(device, verbose=False)
        
        # Load CLM model
        config = AutoConfig.from_pretrained(resolved_path, trust_remote_code=True, torch_dtype=inference_dtype)
        config.is_causal = True
        
        if verbose:
            print(f"Loading CLM model")
        
        model = load_model_with_fallbacks(
            AutoModelForCausalLM, 
            resolved_path,
            config=config,
            inference_dtype=inference_dtype,
            device=device,
            verbose=verbose
        )
        
        if model is None:
            print("Failed to load CLM model, skipping CLM scoring")
        else:
            model.eval()
        
            # Score mutations
            clm_scores = score_mutations_clm(
                sequence,
                mutations,
                model,
                tokenizer,
                batch_size=batch_size,
                window_size=window_size,
                device=device,
                verbose=verbose
            )
            
            # Add to results
            clm_column = f"{model_type}_clm_score"
            results[clm_column] = df['mutant'].map(lambda x: clm_scores.get(x, np.nan))
    
    # If both modes were used, calculate combined score
    if eval_mode == "both" and "mlm" in supported_modes and "clm" in supported_modes:
        if verbose:
            print("Calculating combined score")
        
        mlm_column = f"{model_type}_mlm_score"
        clm_column = f"{model_type}_clm_score"
        
        # Standard normalize each score
        mlm_scores = results[mlm_column].values
        clm_scores = results[clm_column].values
        
        # Remove NaNs for normalization
        valid_indices = ~(np.isnan(mlm_scores) | np.isnan(clm_scores))
        valid_mlm = mlm_scores[valid_indices]
        valid_clm = clm_scores[valid_indices]
        
        if len(valid_mlm) > 0 and len(valid_clm) > 0:
            # Z-score normalization
            norm_mlm = (valid_mlm - np.mean(valid_mlm)) / np.std(valid_mlm)
            norm_clm = (valid_clm - np.mean(valid_clm)) / np.std(valid_clm)
            
            # Create normalized arrays with NaNs preserved
            norm_mlm_full = np.full_like(mlm_scores, np.nan)
            norm_clm_full = np.full_like(clm_scores, np.nan)
            norm_mlm_full[valid_indices] = norm_mlm
            norm_clm_full[valid_indices] = norm_clm
            
            # Calculate average
            combined_scores = (norm_mlm_full + norm_clm_full) / 2
            results[f"{model_type}_score"] = combined_scores
        else:
            if verbose:
                print("Warning: Not enough valid scores to calculate combined score")
    
    # Add results to dataframe
    for column, values in results.items():
        df[column] = values
    
    # Calculate correlations with DMS scores
    correlations = {}
    for column in results.keys():
        valid_data = df.dropna(subset=[column, 'DMS_score'])
        if len(valid_data) > 0:
            correlation, p_value = spearmanr(valid_data['DMS_score'], valid_data[column])
            correlations[column] = {
                'correlation': correlation,
                'p_value': p_value,
                'count': len(valid_data)
            }
            if verbose:
                print(f"{column}: Spearman correlation = {correlation:.4f} (p-value: {p_value:.4e}, n={len(valid_data)})")
        else:
            correlations[column] = {
                'correlation': np.nan,
                'p_value': np.nan,
                'count': 0
            }
            if verbose:
                print(f"{column}: No valid data for correlation")
    
    # Save results to CSV
    if output_path:
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"Results saved to {output_path}")
    
    # Return primary correlation
    if f"{model_type}_score" in correlations:
        return correlations[f"{model_type}_score"]['correlation']
    elif f"{model_type}_mlm_score" in correlations:
        return correlations[f"{model_type}_mlm_score"]['correlation']
    elif f"{model_type}_clm_score" in correlations:
        return correlations[f"{model_type}_clm_score"]['correlation']
    else:
        return np.nan


def test_model(model_type, model_path=None, eval_mode="both", device='cuda', fp16=False, verbose=True):
    """
    Test that the model can be loaded and used correctly.
    
    Args:
        model_type (str): Type of model to use
        model_path (str, optional): Path to local model checkpoint
        eval_mode (str): Evaluation mode - "mlm", "clm", or "both"
        device (str): Device to run the model on
        fp16 (bool): Whether to use half precision
    """
    print(f"Testing {model_type} in {eval_mode} mode...")
    
    # Resolve the model path
    resolved_path = resolve_model_path(model_type, model_path)
    print(f"Resolved model path: {resolved_path}")
    
    # Load tokenizer
    try:
        print(f"Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=True, use_fast=True)
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
        return False
    
    # Model capabilities
    model_capabilities = {
        "proteinglm-100b-int4": ["mlm", "clm"],
        "proteinglm-1b-clm": ["clm"],
        "proteinglm-3b-clm": ["clm"],
        "proteinglm-7b-clm": ["clm"],
        "proteinglm-10b-mlm": ["mlm"],
        "proteinglm-1b-mlm": ["mlm"],
        "proteinglm-3b-mlm": ["mlm"]
    }
    
    supported_modes = model_capabilities[model_type]
    modes_to_test = []
    
    if eval_mode == "both":
        modes_to_test = supported_modes
    elif eval_mode in supported_modes:
        modes_to_test = [eval_mode]
    else:
        print(f"Warning: Model {model_type} does not support {eval_mode} mode. Supported: {supported_modes}")
        return False
    
    # Test sequence
    test_seq = "MILMCQHFSGQFSKYFLAVSSDFCHFVFPIILVSHVNFKQMKRKGF"
    test_mutations = ["M1L", "F10A", "S12T"]
    
    success = True
    inference_dtype = get_inference_dtype(resolved_path, fp16=fp16, verbose=False)

    # Test MLM if needed
    if "mlm" in modes_to_test:
        try:
            print("Testing MLM mode...")

            config = AutoConfig.from_pretrained(resolved_path, trust_remote_code=True, torch_dtype=inference_dtype)
            config.is_causal = False
            config.post_layer_norm = True
            
            model = load_model_with_fallbacks(
                AutoModelForMaskedLM, 
                resolved_path,
                config=config,
                inference_dtype=inference_dtype,
                device=device,
                verbose=True
            )
            
            if model is None:
                print("Failed to load MLM model after trying multiple methods")
                success = False
            else:
                model.eval()

                if verbose:
                    print("Mask token:", tokenizer.mask_token)
                    print("Mask token ID:", tokenizer.mask_token_id)
                
                # Test masked prediction
                masked_seq = test_seq[:5] + tokenizer.mask_token + test_seq[6:]
                if verbose: print("Masked sequence:", masked_seq)
                inputs = tokenizer(masked_seq, add_special_tokens=True, return_tensors="pt").to(device)
                if verbose: print("Tokenized input IDs:", inputs["input_ids"])

                with torch.no_grad():
                    try:
                        inputs = {"input_ids": inputs["input_ids"].cuda(), "attention_mask": inputs["attention_mask"].cuda()}
                        outputs = model(**inputs)
                        
                        # Get mask position
                        mask_positions = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
                        if len(mask_positions) == 0:
                            print("Error: No mask token found in input")
                            success = False
                        else:
                            logits = outputs.logits[0, mask_positions[0]]
                            probs = torch.softmax(logits, dim=-1)
                            
                            # Get top predictions to verify model is working
                            top_k = 5
                            top_probs, top_indices = torch.topk(probs, top_k)
                            
                            print("\nTop predictions for masked token:")
                            for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                                # Try to convert token ID back to amino acid
                                token = tokenizer.convert_ids_to_tokens([idx])[0]
                                print(f"Token: {token}, ID: {idx}, Probability: {prob:.4f}")
                            
                            print("MLM test successful!")
                    
                    except Exception as e:
                        print(f"Error during model forward pass: {e}")
                        # Detailed exception info
                        import traceback
                        traceback.print_exc()
                        success = False
                
                # Only try scoring mutations if the basic test passes
                if success:
                    try:
                        # Score a few mutations as test
                        scores = score_mutations_mlm(
                            test_seq,
                            test_mutations,
                            model,
                            tokenizer,
                            device=device
                        )
                        
                        print(f"Sample MLM mutation scores: {scores}")
                    except Exception as e:
                        print(f"Error scoring mutations: {e}")
                        # This is not a critical failure for the test
                        import traceback
                        traceback.print_exc()
            
        except Exception as e:
            print(f"Error testing MLM mode: {e}")
            import traceback
            traceback.print_exc()
            success = False
    
        del model
        clear_gpu_memory(device, verbose=False)

    # Test CLM if needed
    if "clm" in modes_to_test:
        try:
            print("Testing CLM mode...")
            config = AutoConfig.from_pretrained(resolved_path, trust_remote_code=True, torch_dtype=inference_dtype)
            config.is_causal = True
            
            clear_gpu_memory(device, verbose=False)

            model = load_model_with_fallbacks(
                AutoModelForCausalLM, 
                resolved_path,
                config=config,
                inference_dtype=inference_dtype,
                device=device,
                verbose=True
            )
            
            if model is None:
                print("Failed to load CLM model after trying multiple methods")
                success = False
            else:
                model.eval()
                
                # Test sequence generation (or at least forward pass)
                inputs = tokenizer(test_seq[:10], return_tensors="pt").to(device)
                                
                with torch.no_grad():
                    try:
                        # For CLM, input_ids should exclude the last token
                        input_ids = inputs['input_ids'][:, :-1]
                        attention_mask = None
                        
                        if 'attention_mask' in inputs:
                            attention_mask = inputs['attention_mask'][:, :-1]
                        
                        # Run forward pass
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        
                        # Print a success message
                        print("CLM test successful!")
                        
                        # If you want to check logits
                        if hasattr(outputs, 'logits'):
                            logits = outputs.logits
                            print(f"Logits shape: {logits.shape}")
                    except Exception as e:
                        print(f"Error during CLM forward pass: {e}")
                        import traceback
                        traceback.print_exc()
                        success = False
                
                # Only try scoring mutations if the basic test passes
                if success:
                    try:
                        # Score a few mutations as test
                        scores = score_mutations_clm(
                            test_seq,
                            test_mutations,
                            model,
                            tokenizer,
                            device=device
                        )
                        
                        print(f"Sample CLM mutation scores: {scores}")
                    except Exception as e:
                        print(f"Error scoring mutations with CLM: {e}")
                        import traceback
                        traceback.print_exc()
                        # This is not a critical failure for the test
            
        except Exception as e:
            print(f"Error testing CLM mode: {e}")
            import traceback
            traceback.print_exc()
            success = False
    
    return success


def process_assays_from_file(input_list_csv, base_dms_dir, output_dir, model_type="proteinglm-1b-mlm", batch_size=16,
                           model_path=None, eval_mode="both", dms_index=-1, device='cuda', fp16=False):
    """
    Process multiple assays from a CSV file with DMS_id column.
    
    Args:
        input_list_csv (str): Path to CSV file with list of assays
        base_dms_dir (str): Base directory containing DMS CSV files
        output_dir (str): Directory to save output files
        model_type (str): Type of model to use
        model_path (str, optional): Path to local model checkpoint
        eval_mode (str): Evaluation mode - "mlm", "clm", or "both"
        dms_index (int): Index of DMS to score. If -1, score all DMS assays
        device (str): Device to run the model on
        
    Returns:
        dict: Dictionary of assay IDs and their correlation values
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CSV file with assay list
    assay_list_df = pd.read_csv(input_list_csv)
    
    if 'DMS_id' not in assay_list_df.columns:
        raise ValueError("Input CSV must contain a 'DMS_id' column")
        
    if 'target_seq' not in assay_list_df.columns:
        raise ValueError("Input CSV must contain a 'target_seq' column with protein sequences")
    
    results = {}
    
    # If dms_index is specified and valid, process only that assay
    if dms_index != -1:
        try:
            dms_index = int(dms_index)
            if 0 <= dms_index < len(assay_list_df):
                assay_list_df = assay_list_df.iloc[[dms_index]]
                print(f"Processing only DMS at index {dms_index}: {assay_list_df.iloc[0]['DMS_id']}")
            else:
                print(f"Warning: DMS_index {dms_index} out of range (0-{len(assay_list_df)-1}), processing all assays")
        except ValueError:
            print(f"Warning: Invalid DMS_index '{dms_index}', processing all assays")
    
    # Process each assay
    for idx, row in assay_list_df.iterrows():
        assay = row['DMS_id']
        target_sequence = row['target_seq']
        
        print(f"\n=== Processing assay {assay} ({idx+1}/{len(assay_list_df)}) ===")
        
        # Construct paths
        input_csv = os.path.join(base_dms_dir, f"{assay}.csv")
        output_csv = os.path.join(output_dir, f"{assay}.csv")
        
        # Check if input CSV exists
        if not os.path.exists(input_csv):
            print(f"Error: Input CSV file {input_csv} not found, skipping")
            continue
        
        # Process the assay
        try:
            correlation = process_csv_and_score_mutations(
                input_csv,
                model_type=model_type,
                eval_mode=eval_mode,
                model_path=model_path,
                sequence=target_sequence,
                output_path=output_csv,
                device=device,
                batch_size=batch_size,
                fp16=fp16
            )
            
            results[assay] = correlation
        except Exception as e:
            print(f"Error processing {assay}: {str(e)}")
            results[assay] = np.nan
    
    # Create summary file with correlations
    summary_df = pd.DataFrame({
        'assay': list(results.keys()), 
        'correlation': list(results.values())
    })
    summary_file_path = os.path.join(output_dir, f"correlation_summary_{model_type}_{eval_mode}.csv")
    if os.path.exists(summary_file_path):
        # Append without including headers again
        summary_df.to_csv(summary_file_path, mode='a', header=False, index=False)
    else:
        # Create new file with headers
        summary_df.to_csv(summary_file_path, index=False)
    
    return results

def resolve_model_path(model_type, model_path=None):
    """
    Resolve the full path to the model.
    
    Args:
        model_type (str): Type of model (e.g., "proteinglm-1b-mlm")
        model_path (str, optional): Path to local model or base directory
        
    Returns:
        str: Full path to the model
    """
    if model_path is None:
        # Use model_type directly (will download from HuggingFace)
        return model_type
    
    # Check if model_path is a directory
    if os.path.isdir(model_path):
        # Check if model_path contains the model directly
        if os.path.exists(os.path.join(model_path, "config.json")):
            return model_path
        
        # Check if model_path/{model_type} exists
        model_subdir = os.path.join(model_path, model_type)
        if os.path.exists(model_subdir) and os.path.isdir(model_subdir):
            return model_subdir
    
    print(f"Model path : {model_path}")
    # Either model_path is a direct path to the model or it doesn't exist
    return model_path

def clear_gpu_memory(device, verbose=False):
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
        gc.collect()
        if verbose:
            print("Cleared GPU memory")

def get_inference_dtype(model_path, fp16, verbose=False):
    is_quantized = "int4" in model_path or "int8" in model_path
    if is_quantized:
        if verbose:
            print("Detected quantized model, forcing half precision loading...")
        inference_dtype = torch.float16
    else:
        inference_dtype = torch.float16 if fp16 else torch.float32
    if verbose:
        print(f"Using inference_dtype: {inference_dtype}")
    return inference_dtype

def load_model_with_fallbacks(model_class, model_path, config=None, inference_dtype=torch.float32, device='cuda', verbose=True):
    """
    Load model with various fallback methods if initial loading fails.
    Includes support for multi-GPU distribution and CPU offloading.
    
    Args:
        model_class: Model class (AutoModelForMaskedLM or AutoModelForCausalLM)
        model_path: Path to the model
        config: Model configuration
        device: Device to load the model on
        verbose: Whether to print verbose output
        fp16: Whether to use half precision (default for quantized model)
        
    Returns:
        Loaded model or None if all methods fail
    """
    if verbose:
        print(f"Attempting to load model from {model_path}")
    

    # First, try loading with multi-GPU if available
    try:
        if torch.cuda.device_count() > 1:
            if verbose:
                print(f"Detected {torch.cuda.device_count()} GPUs, trying multi-GPU loading...")
            
            # Get available GPU memory
            max_memory = {}
            for i in range(torch.cuda.device_count()):
                # Allocate 80% of available memory per GPU (converted to GB)
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpu_memory = int(total_memory * 0.8)
                max_memory[i] = f"{gpu_memory}GiB"
            
            # Add CPU memory for potential offloading
            max_memory["cpu"] = "100GiB"
            
            if verbose:
                print(f"Memory map: {max_memory}")
            
            try:
                # Try loading with device_map="auto" for auto distribution
                model = model_class.from_pretrained(
                    model_path,
                    config=config,
                    torch_dtype=inference_dtype,
                    trust_remote_code=True,
                    device_map="auto",
                    max_memory=max_memory
                )
                
                if verbose:
                    print("Successfully loaded model with multi-GPU distribution")
                    if hasattr(model, 'hf_device_map'):
                        print(f"Model device map: {model.hf_device_map}")
                
                return model
            except Exception as e:
                if verbose:
                    print(f"Multi-GPU loading failed: {str(e)}")
    except Exception as e:
        if verbose:
            print(f"Error checking GPU count: {str(e)}")
    
    # Standard loading methods
    methods = [
        # Method 1: Standard loading
        lambda: model_class.from_pretrained(
            model_path, 
            config=config,
            torch_dtype=inference_dtype,
            trust_remote_code=True
        ),
        
        # Method 2: Try with local_files_only=True
        lambda: model_class.from_pretrained(
            model_path, 
            config=config, 
            torch_dtype=inference_dtype, 
            trust_remote_code=True, 
            local_files_only=True
        ),
        
        # Method 3: Try with low_cpu_mem_usage=True
        lambda: model_class.from_pretrained(
            model_path, 
            config=config, 
            torch_dtype=inference_dtype, 
            trust_remote_code=True, 
            low_cpu_mem_usage=True
        ),
        
        # Method 4: Try loading with int8 quantization
        lambda: model_class.from_pretrained(
            model_path, 
            config=config, 
            torch_dtype=inference_dtype,
            trust_remote_code=True,
            load_in_8bit=True
        ),
    ]
    
    # Try each loading method
    errors = []
    for i, method in enumerate(methods):
        try:
            if verbose:
                print(f"Trying loading method {i+1}...")
            model = method()
            if verbose:
                print(f"Successfully loaded model with method {i+1}")
            return model.to(device)
        except Exception as e:
            error_msg = str(e)
            errors.append(f"Method {i+1} failed: {error_msg}")
            if verbose:
                print(f"Method {i+1} failed: {error_msg}")
    
    # Try CPU offloading with accelerate if available
    try:
        import accelerate
        has_accelerate = True
    except ImportError:
        has_accelerate = False
        if verbose:
            print("Warning: accelerate package not found, skipping CPU offloading method")
    
    if has_accelerate:
        try:
            print("Trying CPU offloading with accelerate...")
            
            from accelerate import init_empty_weights, load_checkpoint_and_dispatch
            
            with init_empty_weights():
                model = model_class.from_config(config, trust_remote_code=True)
            
            # Use a device map that offloads to CPU
            model = load_checkpoint_and_dispatch(
                model, 
                model_path, 
                device_map="auto", 
                no_split_module_classes=["xTrimoPGLMBlock"], 
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                offload_folder="offload",
                offload_state_dict=True
            )
            
            if verbose:
                print("Successfully loaded model with CPU offloading")
            
            return model
        except Exception as e:
            error_msg = str(e)
            errors.append(f"CPU offloading failed: {error_msg}")
            if verbose:
                print(f"CPU offloading failed: {error_msg}")
    
    # Try loading with safetensors if available
    try:
        if verbose:
            print("Trying to load with safetensors...")
        
        model = model_class.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
            device_map="auto",
            offload_folder="offload",
            from_tf=False,
            from_safetensors=True
        )
        
        if verbose:
            print("Successfully loaded model with safetensors")
        
        return model
    except Exception as e:
        error_msg = str(e)
        errors.append(f"Safetensors loading failed: {error_msg}")
        if verbose:
            print(f"Safetensors loading failed: {error_msg}")
    
    # If all methods failed, report and return None
    print("ERROR: Failed to load model. Tried the following methods:")
    for error in errors:
        print(f"  - {error}")
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process DMS assays with xtrimopglm model scoring")
    
    parser.add_argument("--model_type", choices=[
                        "proteinglm-100b-int4", 
                        "proteinglm-1b-clm", 
                        "proteinglm-3b-clm", 
                        "proteinglm-7b-clm", 
                        "proteinglm-10b-mlm", 
                        "proteinglm-1b-mlm", 
                        "proteinglm-3b-mlm"], 
                      default="proteinglm-1b-mlm",
                      help="Model type to use for scoring")
    
    parser.add_argument("--model_path", type=str, default="/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/proteinglm-1b-mlm",
                  help="Path to local model or base directory containing models. If a base directory, models will be loaded from {model_path}/{model_type}")
    
    parser.add_argument("--eval_mode", choices=["mlm", "clm", "both"], default="both",
                      help="Evaluation mode: mlm, clm, or both")
    
    parser.add_argument("--reference_csv", type=str, default="/nfs_beijing/kubeflow-user/wanghao/workspace/ai4sci/protein_benchmark_new/protein_benchmark/datasets/DMS_ProteinGym_substitutions/ProteinGym_AF2_structures/DMS_substitutions.csv",
                      help="CSV file with DMS_id and target_seq columns")
    
    parser.add_argument("--dms_dir", type=str, default="/nfs_beijing/kubeflow-user/wanghao/workspace/ai4sci/protein_benchmark_new/protein_benchmark/datasets/DMS_ProteinGym_substitutions",
                      help="Directory containing DMS CSV files")
    
    parser.add_argument("--output_dir", type=str, default="./pglm_test",
                      help="Directory to save output files")
    
    parser.add_argument("--DMS_index", required=False, default=-1,
                      help="Index of DMS to score. If not provided, score all DMS assays")
    
    parser.add_argument("--batch_size", required=False, default=8,
                      help="Batch size to use for scoring")
    
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to run the model on (cuda or cpu)")
    
    parser.add_argument("--test", action="store_true", 
                      help="Run in test mode to verify model loading and scoring")
    
    parser.add_argument("--fp16", action="store_true", 
                      help="Use half precision for model inference")
    
    args = parser.parse_args()
    
    # Check if we're in test mode
    if args.test:
        success = test_model(
            model_type=args.model_type,
            model_path=args.model_path,
            eval_mode=args.eval_mode,
            device=args.device,
            fp16=args.fp16,
            verbose=True
        )
        if success:
            print("Model testing successful!")
        else:
            print("Model testing failed.")
        
        exit(0 if success else 1)
    
    # Verify required arguments for normal operation
    if args.reference_csv is None or args.dms_dir is None or args.output_dir is None:
        parser.error("--reference_csv, --dms_dir, and --output_dir are required when not in test mode")
    
    # Process all assays from the input list
    results = process_assays_from_file(
        input_list_csv=args.reference_csv,
        base_dms_dir=args.dms_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        batch_size=args.batch_size,
        model_path=args.model_path,
        eval_mode=args.eval_mode,
        dms_index=args.DMS_index,
        device=args.device,
        fp16=args.fp16
    )
    
    # Print summary of results
    print("\n=== Summary of Results ===")
    for assay, correlation in results.items():
        if not np.isnan(correlation):
            print(f"{assay}: Spearman correlation = {correlation:.4f}")
        else:
            print(f"{assay}: Failed to calculate correlation")
