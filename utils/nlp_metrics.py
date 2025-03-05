import re
import nltk  # Only needed if you want Self-BLEU per sample
from rouge_score import rouge_scorer
from bert_score import score
import Levenshtein
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def basic_tokenize(text: str):
    """
    Lowercase and split text into a list of tokens.
    Removes punctuation.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    return tokens


def normalize_for_cer(s: str) -> str:
    """
    Mimics jiwer-like text normalization for CER:
    1. Strip leading/trailing whitespace
    2. Convert multiple spaces to a single space
    3. (Optional) Lowercase
    """
    s = s.strip()
    s = re.sub(r"\s+", " ", s)  # unify multiple spaces
    s = s.lower()               # match your existing lowercase approach, if desired
    return s

def compute_cer(reference: str, prediction: str) -> float:
    """
    Computes a jiwer-like Character Error Rate (CER):
    1. Normalize and unify spaces
    2. Remove spaces entirely for the distance calculation (optional, but often done)
    3. Divide edit distance by length of the reference
    """
    ref_norm = normalize_for_cer(reference)
    pred_norm = normalize_for_cer(prediction)
    
    # Remove all spaces so "the cat" -> "thecat"
    ref_chars = ref_norm.replace(" ", "")
    pred_chars = pred_norm.replace(" ", "")

    # Compute character-level Levenshtein distance
    distance = Levenshtein.distance(ref_chars, pred_chars)

    # Avoid division by zero
    return distance / len(ref_chars) if len(ref_chars) > 0 else 0.0


def compute_rouge_1_f(reference_tokens: list[str], pred_tokens: list[str]) -> float:
    """
    Computes ROUGE-1 F-measure (unigram overlap). Expects tokenized input (lists of tokens).
    Measures quality of automatic summarization.
    """
    # Re-join tokens into strings for the scorer.
    ref_text = " ".join(reference_tokens)
    pred_text = " ".join(pred_tokens)

    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = scorer.score(ref_text, pred_text)
    return scores["rouge1"].fmeasure


def compute_sentence_bleu(reference_tokens: list[str], pred_tokens: list[str]) -> float:
    """
    Computes the sentence-level BLEU score for a single prediction-reference pair.
    Uses nltk's sentence_bleu which computes the BLEU score for the given tokens.
    """
    smoother = SmoothingFunction().method1
    
    return sentence_bleu(
        [reference_tokens],
        pred_tokens,
        smoothing_function=smoother
    )

def compute_self_bleu(generated_texts: list[str]) -> float:
    """
    Given a list of generated strings, compute the average BLEU
    of each string against all others. Measures diversity.
    """
    all_scores = []
    for i, text_i in enumerate(generated_texts):
        pred_tokens = basic_tokenize(text_i)
        # Treat every other text as a reference.
        references = [
            basic_tokenize(text_j) for j, text_j in enumerate(generated_texts) if j != i
        ]
        scores = [compute_sentence_bleu(ref, pred_tokens) for ref in references]
        avg_score = sum(scores) / len(scores) if scores else 0
        all_scores.append(avg_score)
    return sum(all_scores) / len(all_scores) if all_scores else 0

def nlp_metrics(
    predictions: list[str],
    references: list[str],
    use_normalization: bool = True,
    use_bleu: bool = True,
    use_rouge_f: bool = True,
    use_bert_score: bool = True,
    use_cer: bool = True,
    use_self_bleu: bool = True,
    bert_model_type: str = "bert-base-uncased",
    batch_size: int = 256,
) -> dict:
    """
    Computes various NLP metrics over a batch of prediction-reference pairs (same length).

    Before any metric is computed, this function filters out any prediction-reference pair
    where either the prediction or the reference is an empty string.
    
    :param predictions: List of predicted strings of length N.
    :param references: List of reference strings of length N.
    :param use_normalization: If True, applies basic_tokenize for metrics that benefit from it.
    :param use_bleu: If True, computes the average sentence-level BLEU across samples using nltk's sentence_bleu.
    :param use_rouge_f: If True, computes average ROUGE-1 F1 across samples.
    :param use_bert_score: If True, computes BERTScore F1 in one batch pass.
    :param use_cer: If True, computes Character Error Rate (CER) across samples.
    :param use_self_bleu: If True, computes Self-BLEU over all predictions (measures diversity).
    :param bert_model_type: Model to use for BERTScore (default is "bert-base-uncased").
    :param batch_size: Batch size to use for BERTScore computation.
    :return: Dictionary with average metric values.
    """
    # Filter out pairs with empty predictions or references
    filtered = [(p, r) for p, r in zip(predictions, references) if p.strip() != "" and r.strip() != ""]
    if not filtered:
        return {
            "bleu": 0,
            "rouge_f": 0,
            "bert_score": 0,
            "cer": 0,
            "self_bleu": 0,
        }
    predictions, references = zip(*filtered)
    predictions = list(predictions)
    references = list(references)

    metrics = {}
    n = len(predictions)

    # For sample-level metrics (ROUGE, CER, and sentence BLEU) we accumulate per-sample.
    rouge_f_scores = []
    cer_scores = []
    bleu_scores = []

    # Optional tokenization for sample-level metrics
    if use_normalization:
        tokenized_refs = [basic_tokenize(r) for r in references]
        tokenized_preds = [basic_tokenize(p) for p in predictions]
    else:
        tokenized_refs = [r.split() for r in references]
        tokenized_preds = [p.split() for p in predictions]

    # 1) Sentence-level BLEU: compute average BLEU per prediction-reference pair.
    if use_bleu:
        for pred_tokens, ref_tokens in zip(tokenized_preds, tokenized_refs):
            bleu_scores.append(compute_sentence_bleu(ref_tokens, pred_tokens))
        metrics["bleu"] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

    # 2) ROUGE-1 F: compute average across samples.
    if use_rouge_f:
        for ref_toks, pred_toks in zip(tokenized_refs, tokenized_preds):
            rouge_f_scores.append(compute_rouge_1_f(ref_toks, pred_toks))
        metrics["rouge_f"] = sum(rouge_f_scores) / len(rouge_f_scores) if rouge_f_scores else 0

    # 3) BERTScore in one batch (avoids per-sample overhead)
    if use_bert_score:
        from bert_score import score  # in case not imported yet
        P, R, F1 = score(
            cands=predictions,
            refs=references,
            model_type=bert_model_type,
            lang="en",
            use_fast_tokenizer=True,
        )
        avg_f1 = float(sum(F1) / len(F1))
        metrics["bert_score"] = avg_f1

    # 4) CER: compute average across samples.
    if use_cer:
        for ref, pred in zip(references, predictions):
            cer_scores.append(compute_cer(ref, pred))
        metrics["cer"] = sum(cer_scores) / len(cer_scores) if cer_scores else 0

    # 5) Self-BLEU: measure diversity among predictions.
    if use_self_bleu:
        metrics["self_bleu"] = compute_self_bleu(predictions)

    return metrics
