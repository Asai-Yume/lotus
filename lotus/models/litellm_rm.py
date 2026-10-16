import time
from typing import Any

import numpy as np
from litellm import embedding
from litellm.types.utils import EmbeddingResponse
from numpy.typing import NDArray
from tqdm import tqdm

from lotus.dtype_extensions import convert_to_base_data
from lotus.models.rm import RM


class LiteLLMRM(RM):
    """
    A retrieval model based on LiteLLM embedding models.

    This class provides functionality to generate embeddings for documents using
    various embedding models supported by LiteLLM. It supports batch processing
    and optional text truncation for efficient embedding generation.

    Attributes:
        model (str): Name of the embedding model to use.
        max_batch_size (int): Maximum batch size for embedding requests.
        truncate_limit (int | None): Maximum character limit for text truncation.
        max_retries (int): Number of retries for transient provider failures after
            the first physical attempt (defaults to 0 = no retries).
        retry_backoff_seconds (float): Base exponential backoff between retries
            (capped at 30 seconds). Defaults to 2.0.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        max_batch_size: int = 64,
        truncate_limit: int | None = None,
        max_retries: int = 0,
        retry_backoff_seconds: float = 2.0,
    ) -> None:
        """
        Initialize the LiteLLMRM retrieval model.

        Args:
            model: Name of the embedding model to use. Defaults to "text-embedding-3-small".
            max_batch_size: Maximum batch size for embedding requests. Defaults to 64.
            truncate_limit: Maximum character limit for text truncation.
                           If None, no truncation is applied. Defaults to None.
            max_retries: Number of retries for transient provider failures after
                the first physical attempt. Defaults to 0 (no retries).
            retry_backoff_seconds: Base exponential backoff between retries,
                capped at 30 seconds. Defaults to 2.0.
        """
        super().__init__()
        self.model: str = model
        self.max_batch_size: int = max_batch_size
        self.truncate_limit: int | None = truncate_limit
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0.")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be >= 0.")
        self.max_retries: int = max_retries
        self.retry_backoff_seconds: float = retry_backoff_seconds
        self.retry_max_backoff_seconds: float = 30.0

    def _embed(self, docs: list[str]) -> NDArray[np.float64]:
        """
        Generate embeddings for a list of documents.

        This method processes documents in batches to generate embeddings using
        the specified embedding model. It supports optional text truncation and
        shows progress with a progress bar.

        Args:
            docs: List of document strings to embed.

        Returns:
            NDArray[np.float64]: Array of embeddings with shape (num_docs, embedding_dim).

        Raises:
            Exception: If the embedding API request fails.
        """
        all_embeddings = []
        for i in tqdm(range(0, len(docs), self.max_batch_size)):
            batch = docs[i : i + self.max_batch_size]
            if self.truncate_limit:
                batch = [doc[: self.truncate_limit] for doc in batch]
            _batch = convert_to_base_data(batch)
            embeddings = self._embed_batch_with_retries(_batch, expected_count=len(batch))
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    def _embed_batch_with_retries(self, api_batch: list[Any], expected_count: int) -> NDArray[np.float64]:
        """Embed one batch with bounded transient retries (model layer).

        A malformed response (missing/empty data or a wrong result count) is
        treated as a transient provider glitch and retried while retries remain;
        otherwise it fails with a clear error. Retries never change the returned
        embedding values.
        """
        from lotus.models.lm import is_retryable_exception

        attempt = 0
        while True:
            malformed: Exception | None = None
            try:
                response: EmbeddingResponse = embedding(model=self.model, input=api_batch)
                data = getattr(response, "data", None)
                if data is None or len(data) != expected_count:
                    actual = 0 if data is None else len(data)
                    malformed = RuntimeError(
                        f"Embedding response returned {actual} results for {expected_count} inputs "
                        f"(model={self.model})."
                    )
                    raise malformed
                return np.array([d["embedding"] for d in data])
            except RuntimeError:
                if malformed is None or attempt >= self.max_retries:
                    raise
            except Exception as exc:
                if attempt >= self.max_retries or not is_retryable_exception(exc):
                    raise
            attempt += 1
            time.sleep(
                min(self.retry_max_backoff_seconds, self.retry_backoff_seconds * (2 ** max(0, attempt - 1)))
            )
