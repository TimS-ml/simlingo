"""Evaluation metrics suite for language model outputs.

This module provides comprehensive evaluation metrics for SimLingo's language
generation capabilities (QA and commentary). It computes:

1. Exact Match Accuracy: Binary correct/incorrect for factual answers
2. NLG Metrics: BLEU, ROUGE-L, CIDEr, METEOR, SPICE for fluency and content
3. GPT-4 Scoring: LLM-based evaluation for semantic correctness

The evaluation suite handles large datasets by chunking them for memory efficiency
and supports parallel processing for GPT-4 evaluation.

Usage:
    python eval_metrics.py --root_path1 predictions.json

Input Format:
    JSON file with list of [prediction, ground_truth] pairs

Output:
    JSON file with computed metrics (saved alongside input file)
"""

import re
import argparse
import json
import numpy as np
import torch.nn as nn
import language_evaluation
from multiprocessing import Pool
from tqdm import tqdm

import sys
sys.path.append(".")
from utils.gpt_eval import *


class evaluation_suit():
    """Comprehensive evaluation suite for language generation tasks.

    This class accumulates predictions and ground truths, then computes
    multiple complementary metrics:
    - Accuracy: Exact string matching (for factual QA)
    - Language metrics: Traditional NLG metrics (BLEU, ROUGE, etc.)
    - GPT scoring: LLM-based semantic evaluation

    The class handles preprocessing (removing special tokens) and chunking
    for memory-efficient evaluation on large datasets.

    Attributes:
        language_eval: CocoEvaluator for computing NLG metrics
        GPT: List of (answer, GT) tuples for GPT-4 evaluation
        accuracy: Dict with 'answer' and 'GT' lists for accuracy computation
        language: Dict with 'answer' and 'GT' lists for NLG metrics
    """
    def __init__(self):
        """Initialize evaluation suite with metric computers.

        Sets up:
        - CocoEvaluator for NLG metrics (BLEU, ROUGE-L, CIDEr, METEOR, SPICE)
        - Storage for predictions and ground truths
        """
        self.language_eval = language_evaluation.CocoEvaluator(
            coco_types=["BLEU", "ROUGE_L", "CIDEr", "METEOR", "SPICE"]
        )
        self.GPT = []  # For GPT-4 evaluation
        self.accuracy = {"answer": [], "GT": []}  # For exact match
        self.language = {"answer": [], "GT": []}  # For NLG metrics

    def eval_acc(self):
        """Compute exact match accuracy.

        Returns:
            float: Accuracy score (fraction of exact matches) in range [0, 1]

        This is a strict metric useful for factual QA where answers must be
        exactly correct (e.g., "yes"/"no" questions).
        """
        scores = []
        for i in tqdm(range(len(self.accuracy["answer"]))):
            answer = self.accuracy["answer"][i]
            GT = self.accuracy["GT"][i]
            # Binary score: 1.0 for exact match, 0.0 otherwise
            if answer == GT:
                scores.append(1.0)
            else:
                scores.append(0.0)

        # Return average accuracy across all samples
        scores = sum(scores) / len(scores)
        return scores

    def eval_chatGPT(self, data):
        """Evaluate predictions using GPT-4 as a judge.

        This uses GPT-4 to score answers on semantic correctness, handling
        paraphrases and equivalent answers that exact match would miss.

        Args:
            data: List of (prediction, ground_truth) tuples

        Returns:
            float: Average GPT-4 score across all valid samples

        The function:
        - Uses parallel processing (16 workers) for speed
        - Filters out failed API calls (scored as -1)
        - Reports number of invalid samples
        """
        # Parallel GPT-4 API calls (16 workers - adjust based on rate limits)
        with Pool(16) as p:
            scores_all = p.map(gpt_forward, data)

        # Filter out failed API calls (returned as -1)
        scores = [x for x in scores_all if x != -1]
        delted = len(scores_all) - len(scores)
        print(f"Deleted {delted} invalid samples")

        # Convert to float and average
        scores = list(map(float, scores))
        scores = sum(scores) / len(scores)
        return scores

    def eval_language(self):
        """Compute natural language generation metrics.

        Computes BLEU, ROUGE-L, CIDEr, METEOR, and SPICE scores using the
        language_evaluation library (COCO evaluation toolkit).

        Returns:
            dict: Metric scores with keys like 'val/BLEU', 'val/ROUGE_L', etc.

        The function implements chunking for memory efficiency:
        - Chunks data into 500-sample batches
        - Computes metrics per chunk
        - Combines with weighted averaging
        - Returns prefixed keys ('val/') for logging

        Note:
            SPICE computation can be memory-intensive for large batches,
            hence the chunking strategy.
        """
        answer = self.language["answer"]
        GT = self.language["GT"]
        chunk_size = 500  # Process in chunks to avoid memory issues
        n_total = len(answer)

        # Fast path: single chunk (no need to split)
        if n_total <= chunk_size:
            results = self.language_eval.run_evaluation(answer, GT)
            return {f"val/{k}": float(v) for k, v in results.items()}

        # Multi-chunk path: split, evaluate, and combine with weighted averaging
        results_accumulator = {}
        total_items = 0

        for i in range(0, n_total, chunk_size):
            # Extract chunk
            answer_split = answer[i:i + chunk_size]
            GT_split = GT[i:i + chunk_size]
            chunk_len = len(answer_split)
            total_items += chunk_len

            # Evaluate chunk
            results_gen = self.language_eval.run_evaluation(answer_split, GT_split)

            # Accumulate weighted sums (weight by chunk size)
            for k, v in results_gen.items():
                results_accumulator[k] = results_accumulator.get(k, 0.0) + float(v) * chunk_len

        # Compute weighted mean over all items
        results_gen_dict = {f"val/{k}": v_sum / total_items for k, v_sum in results_accumulator.items()}
        return results_gen_dict

    def forward(self, answer, GT):
        """Add a prediction-ground truth pair to the evaluation queue.

        This method preprocesses the strings (removes special tokens) and
        stores them for later batch evaluation.

        Args:
            answer: Model's predicted answer
            GT: Ground truth answer

        The preprocessing removes:
        - 'A: ' prefix (answer format marker)
        - ' <|im_end|>' suffix (end-of-message token)
        """
        # Remove answer prefix and end-of-message token
        answer = answer.replace('A: ', '')
        GT = GT.replace('A: ', '')
        answer = answer.replace(' <|im_end|>', '')
        GT = GT.replace(' <|im_end|>', '')

        # Store for all three evaluation methods
        self.accuracy["answer"].append(answer)
        self.accuracy["GT"].append(GT)
        self.GPT.append((answer, GT))
        self.language["GT"].append(GT)
        self.language["answer"].append(answer)

    def evaluation(self):
        """Run all evaluation metrics and aggregate results.

        This method runs all three evaluation approaches:
        1. Exact match accuracy
        2. GPT-4 semantic scoring
        3. NLG metrics (BLEU, ROUGE, CIDEr, METEOR, SPICE)

        Returns:
            dict: Dictionary with keys 'accuracy', 'chatgpt', 'language'
                  Each contains the corresponding metric scores

        Error handling:
        - Each metric is wrapped in try-except
        - Failed metrics return default values (0.0 or {})
        - Errors are printed but don't stop other metrics
        """
        print("evaluation start!")
        scores = {}

        # Exact match accuracy
        try:
            print("accuracy evaluation")
            scores["accuracy"] = self.eval_acc()
        except:
            print("Error in accuracy evaluation")
            scores["accuracy"] = 0.0

        # GPT-4 scoring (can fail due to API issues)
        try:
            print("chatGPT evaluation")
            scores["chatgpt"] = self.eval_chatGPT(self.GPT)
        except:
            print("Error in chatGPT evaluation")
            scores["chatgpt"] = 0.0

        # NLG metrics
        try:
            print("language evaluation")
            scores["language"] = self.eval_language()
        except:
            print("Error in language evaluation")
            scores["language"] = {}

        return scores


if __name__ == '__main__':
    """Command-line interface for metric computation.

    Loads predictions from JSON file, computes all metrics, and saves results.

    Expected JSON format:
        [[pred1, gt1], [pred2, gt2], ...]

    Output file:
        Input filename with '_metrics_gpt-4o-2024-08-06.json' suffix
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument(
        '--root_path1',
        type=str,
        default="outputs/simlingo/predictions/language_preds_cot_rank_0.json",
        help='path to prediction file'
    )
    args = parser.parse_args()

    # Load predictions from JSON
    with open(args.root_path1, 'r') as f:
        pred_file = json.load(f)

    # Extract predictions and ground truths
    # Format: [[pred, gt], [pred, gt], ...]
    gt = [preds[1].replace('A: ','') for preds in pred_file]
    pred = [preds[0].replace('A: ','') for preds in pred_file]

    print("Number of predictions: ", len(pred))
    print("Number of ground truths: ", len(gt))

    # Initialize evaluation suite
    evaluation = evaluation_suit()

    # Add all predictions to evaluation queue
    for i in range(len(pred)):
        evaluation.forward(pred[i], gt[i])

    # Compute all metrics
    output = evaluation.evaluation()

    # Print results
    print("accuracy score: ", output["accuracy"])
    print("chatgpt score: ", output["chatgpt"])
    print("language score: ", output["language"])

    # Save results to JSON file
    save_path = args.root_path1.replace(".json", "_metrics_gpt-4o-2024-08-06.json")
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"Results saved to: {save_path}")