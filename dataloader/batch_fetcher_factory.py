from .audio_batch import AudioBatchFetcher
from .audio_text_batch import AudioTextBatchFetcher

# String names have to match recording types
BATCHTYPES = {
    "audio": AudioBatchFetcher,
    "audiotext": AudioTextBatchFetcher,
}


class BatchFetcherFactory:
    @staticmethod
    def create_batch_fetcher(batch_type: str, **kwargs):
        if batch_type not in BATCHTYPES:
            raise ValueError(f"Batch type {batch_type} not found")
        return BATCHTYPES[batch_type].remote(**kwargs)
