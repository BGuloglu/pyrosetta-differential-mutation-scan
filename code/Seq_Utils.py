import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from transformers import AutoTokenizer, EsmForMaskedLM
import torch

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def calculate_log_likelihoods(
    protein_sequence: str, model_name: str = "facebook/esm2_t30_150M_UR50D"
) -> np.ndarray:
    """Takes a protein sequence (single-letter) and runs it through a specified ESM
    model. For each position, the wt probability and the probability of all other
    residues is calculated, and the log likelihood ratio (LLR) is calculated using:

    LLR = log(p(x_mut)) - log(p(x_wt))

    The function returns an array giving the LLR for each mutation in each position.
    The possible models are found at https://huggingface.co/facebook. Larger models will
    require more memory and take longer to run.

    This function is adapted from this blog post and modified so that the entire
    sequence is considered and that the model variable is exposed to the user:
    https://huggingface.co/blog/AmelieSchreiber/mutation-scoring

    Args:
        protein_sequence (str): Protein sequence in one-letter code.
        model_name (str, optional): Name of EMS model to use. Defaults to
                                    "facebook/esm2_t30_150M_UR50D".

    Returns:
        np.ndarray: (20, n) array of LLRs, where n is the length of the input sequence.
    """
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)

    # Tokenize the input sequence
    input_ids = tokenizer.encode(protein_sequence, return_tensors="pt")
    sequence_length = input_ids.shape[1] - 2  # Excluding the special tokens

    # Initialize heatmap
    log_likelihoods = np.zeros((20, sequence_length))

    # Calculate LLRs for each position and amino acid
    for position in range(1, sequence_length + 1):
        # Mask the target position
        masked_input_ids = input_ids.clone()
        masked_input_ids[0, position] = tokenizer.mask_token_id

        # Get logits for the masked token
        with torch.no_grad():
            logits = model(masked_input_ids).logits

        # Calculate log probabilities
        probabilities = torch.nn.functional.softmax(logits[0, position], dim=0)
        log_probabilities = torch.log(probabilities)

        # Get the log probability of the wild-type residue
        wt_residue = input_ids[0, position].item()
        log_prob_wt = log_probabilities[wt_residue].item()

        # Calculate LLR for each variant
        for i, amino_acid in enumerate(AMINO_ACIDS):
            log_prob_mt = log_probabilities[
                tokenizer.convert_tokens_to_ids(amino_acid)
            ].item()
            log_likelihoods[i, position - 1] = log_prob_mt - log_prob_wt

    return log_likelihoods


def save_esm_results(
    log_likelihoods: np.ndarray,
    output_path: str,
    protein_sequence: str,
    notebook: bool = False,
) -> None:
    """Saves log likelihood rations (LLRs) both as a heatmap and as a .npy file.

    Args:
        log_likelihoods (np.ndarray): (20, n) array of LLRs, where n is the length of
                                      the input sequence
        output_path (str): Path to output directory
        protein_sequence (str): Protein sequence in 1-letter code
        notebook (bool, optional): Whether to also display the plot. Useful if running
                                   in a notebook. Defaults to False.

    Returns:
        None
    """

    one_hot_encoded = np.array([AMINO_ACIDS.index(res) for res in protein_sequence])
    one_hot_encoded = np.eye(len(AMINO_ACIDS))[one_hot_encoded]
    annotation_matrix = np.where(one_hot_encoded == 1, "•", "").T

    # Create and save the plot
    f = plt.figure(figsize=[20, 5])
    ax = sns.heatmap(
        log_likelihoods,
        center=0.00,
        cmap="vlag",
        annot=annotation_matrix,
        fmt="",
        cbar_kws={"label": "Log Likelihood Relative to WT"},
        annot_kws={"color": "black"},
    )
    ax.set_xlabel("Position")
    ax.set_ylabel("Amino Acid")
    ax.set_yticklabels(AMINO_ACIDS)
    f.savefig(os.path.join(output_path, "esm_probabilities.pdf"), bbox_inches="tight")

    # Also show the plot if running in a notebook
    if notebook:
        plt.show()

    # Save the data as well for future use
    np.save(os.path.join(output_path, "esm_probabilities.npy"), log_likelihoods)

    return None


def find_acceptable_mutations(
    protein_sequence: str, log_likelihoods: np.ndarray, cutoff: float = -0.5
) -> Dict[int, list]:
    """Uses log likelihood rations to suggest position-specific mutans that are unlikely
       to disupt function. For each position, all variants that have a LLR compared to
       WT that are within a specific threshold are selected.

    Args:
        protein_seq (str): Protein sequence in one-letter code.
        log_likelihoods (np.ndarray): (20,n) array of LLRs, where n is the length of the
                                      input sequence.
        cutoff (float, optional): Cutoff for LLR for a mutation to be acceptable. Values
                                  above 0 will only consider mutations that are expected
                                  to be "better" than the WT residue. Defaults to -0.5.

    Returns:
        Dict[int, list]: Dictionary where each acceptable mutations based on the LLR
                         cutoff are given for each position of the protein sequence
                         NB: the sequence residue id is 1-indexed.
    """

    acceptable_mutations = {}

    # For each position
    for i in range(len(protein_sequence)):
        position_likelihoods = log_likelihoods[:, i]  # Get position LLRs
        amino_acids = np.array(AMINO_ACIDS)  # List of amino acids
        # Find residues which have an acceptable LLR compared to WT
        likely_residues = amino_acids[np.where(position_likelihoods >= cutoff)[0]]
        # Remove the WT residue since we don't want to mutate to it
        likely_mutants = [res for res in likely_residues if res != protein_sequence[i]]
        acceptable_mutations[i + 1] = likely_mutants

    return acceptable_mutations
