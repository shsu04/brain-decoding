from .audio_batch import AudioBatchFetcher

# String names have to match recording types
BATCHTYPES = {
    "audio": AudioBatchFetcher,
}


class BatchFetcherFactory:
    @staticmethod
    def create_batch_fetcher(batch_type: str, **kwargs):
        if batch_type not in BATCHTYPES:
            raise ValueError(f"Batch type {batch_type} not found")
        return BATCHTYPES[batch_type].remote(**kwargs)
