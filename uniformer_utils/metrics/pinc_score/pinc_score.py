# This code has been adapted from the GitHub repository 'https://github.com/potamides/uniformers'
# Portions of the original code have been modified to fit the specific requirements
# of this project. Credit goes to the original authors for their contributions.

import datasets.metric
from datasets import Features, Value
from datasets.info import MetricInfo
from statistics import mean
from transformers.utils import logging
import random
from nltk.util import ngrams
from collections import Counter
from sentence_transformers import SentenceTransformer, util

logger = logging.get_logger("transformers")


class PINC(datasets.metric.Metric):
    """Score consonants and vowels in the masked words of a quatrain"""

    def __init__(
        self,
        batch_size=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    def _info(self):
        return MetricInfo(
            description=str(self.__doc__),
            citation="",
            features=Features(
                {
                    "original": Value("string"),
                    "predicted_paraphrase": Value("string"),
                }
            ),
        )

    def pinc_score(self, original, paraphrase, n=4):
        """Calculate the PINC score for the given original and paraphrase texts."""

        sum = 0
        index = 0
        for i in range(1, n+1):
            original_ngrams = ngrams(original, i)
            paraphrase_ngrams = ngrams(paraphrase, i)

            # calculare intersection of ngrams between original and paraphrase
            original_counter = Counter(original_ngrams)
            paraphrase_counter = Counter(paraphrase_ngrams)

            if original_counter and paraphrase_counter:
                index += 1
                intersection_counter = original_counter & paraphrase_counter
                try:
                    sum += 1 - len(intersection_counter) / \
                        len(paraphrase_counter)
                except ZeroDivisionError:
                    print(original)
                    print(paraphrase)
                    sum += 1

            # very different cases
            if index == 0:
                if original_counter:
                    return 1
                else:
                    return 0

        return sum / index

    def _compute(
        self,
        original,
        predicted_paraphrase,
    ):
        scores = list()
        similarity_scores = list()
        normalized_pinc_scores = list()

        for i in range(0, len(original)):
            scores.append(self.pinc_score(
                original[i], predicted_paraphrase[i], n=4))

            original_embedding = self.model.encode(
                original[i], convert_to_tensor=True)
            predicted_embedding = self.model.encode(
                predicted_paraphrase[i], convert_to_tensor=True)

            cosine_scores = util.pytorch_cos_sim(
                original_embedding, predicted_embedding)
            similarity_scores.append(cosine_scores.item())

            normalized_pinc_scores.append(scores[i] * cosine_scores.item())

        # get a random number between 0 and len(predicted_words)
        i = random.randint(0, len(original) - 1)
        logger.info(
            f"Sample: original sentence is {original[i]} and predicted sentence is {predicted_paraphrase[i]}")

        output_dict = {
            "pinc_score": mean(scores),
            "cosine_similarity": mean(similarity_scores),
            "normalized_pinc_score": mean(normalized_pinc_scores)
        }
        return output_dict
