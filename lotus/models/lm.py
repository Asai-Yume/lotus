import hashlib
import logging
import math
import time
import warnings
from collections import deque
from typing import Any

import numpy as np
from litellm import batch_completion
from litellm.exceptions import AuthenticationError
from litellm.types.utils import ChatCompletionTokenLogprob, ChoiceLogprobs, Choices, ModelResponse
from litellm.utils import decode, encode, token_counter

# ``supports_reasoning`` moved around / was added in newer litellm. Import defensively so
# `import lotus` never hard-fails on older litellm — fall back to the top-level symbol,
# then to None (treated as "not a reasoning model") if unavailable.
try:
    from litellm.utils import supports_reasoning as _supports_reasoning
except ImportError:  # pragma: no cover - depends on litellm version
    try:
        from litellm import supports_reasoning as _supports_reasoning
    except ImportError:  # pragma: no cover
        _supports_reasoning = None  # type: ignore[assignment]
from openai._exceptions import OpenAIError
from pydantic import BaseModel
from tokenizers import Tokenizer
from tqdm import tqdm

import lotus
from lotus.cache import CacheFactory
from lotus.pricing import calculate_cost_from_response
from lotus.types import (
    LMOutput,
    LMStats,
    LogprobsForCascade,
    LogprobsForFilterCascade,
    LotusRetryExhaustedError,
    LotusUsageLimitException,
    UsageLimit,
)

logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)

# Suppress Pydantic serialization warnings emitted by litellm when its ModelResponse
# Choices/Message types don't exactly match the expected StreamingChoices schema.
# This is a known upstream litellm issue and the warnings are harmless — the values are
# still serialized correctly.
#
# We do this two ways because a plain filter isn't reliable across environments: the
# warning is emitted from worker threads / litellm's frame (so a module-scoped filter
# leaks), and other libraries can reset ``warnings.filters`` after we set it. So we both
# register the filter AND install a ``showwarning`` shim that drops exactly these and
# delegates everything else — the shim survives filter-list resets.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

_PYDANTIC_SERIALIZER_WARNING = "Pydantic serializer warnings"
_prev_showwarning = warnings.showwarning


def _drop_pydantic_serializer_warnings(message, category, filename, lineno, file=None, line=None):
    if issubclass(category, UserWarning) and str(message).startswith(_PYDANTIC_SERIALIZER_WARNING):
        return
    return _prev_showwarning(message, category, filename, lineno, file=file, line=line)


# Install once (idempotent — don't wrap our own shim if the module is re-imported).
if getattr(warnings.showwarning, "__name__", "") != "_drop_pydantic_serializer_warnings":
    warnings.showwarning = _drop_pydantic_serializer_warnings

# Default completion-token budgets. Reasoning models (gpt-5, o-series, ...) spend
# hidden reasoning tokens from the same max_completion_tokens budget as the visible
# answer, so they need a much larger default: with the standard 512-token budget a
# reasoning model can exhaust the entire budget before emitting any visible text,
# returning an empty completion that operators silently coerce to their default
# output (see issue #255).
DEFAULT_MAX_TOKENS = 512
DEFAULT_REASONING_MAX_TOKENS = 8192

# HTTP status codes treated as transient provider failures when no stronger
# structural signal (exception class / provider error metadata) is available.
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

# Provider error types (OpenRouter `error.metadata.error_type`) treated as transient.
_RETRYABLE_PROVIDER_ERROR_TYPES = frozenset(
    {
        "provider_unavailable",
        "upstream_error",
        "stream_error",
        "bad_gateway",
        "service_unavailable",
        "timeout",
        "rate_limited",
        "internal_error",
    }
)


def _exception_status_code(exc: BaseException) -> int | None:
    """Best-effort extraction of an HTTP status code from a provider exception."""
    try:
        status = getattr(exc, "status_code", None)
    except Exception:
        status = None
    if status is None:
        try:
            status = getattr(getattr(exc, "response", None), "status_code", None)
        except Exception:
            status = None
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def is_retryable_exception(exc: BaseException) -> bool:
    """Structural, class-first classification of a provider exception.

    Returns True for transient provider/network failures that are safe to retry
    (bounded by the caller's max_retries), False for permanent
    authorization/billing/configuration failures. This deliberately avoids
    free-text substring matching; status codes are only a fallback for
    exception classes litellm does not map.
    """
    from litellm.exceptions import (
        APIConnectionError,
        APIError,
        AuthenticationError,
        BadGatewayError,
        BadRequestError,
        BudgetExceededError,
        ContentPolicyViolationError,
        ContextWindowExceededError,
        GuardrailRaisedException,
        InternalServerError,
        InvalidRequestError,
        NotFoundError,
        PermissionDeniedError,
        RateLimitError,
        ServiceUnavailableError,
        Timeout,
        UnprocessableEntityError,
        UnsupportedParamsError,
    )
    import openai

    # Transient classes first: they take precedence over generic status checks.
    if isinstance(
        exc,
        (
            Timeout,
            RateLimitError,
            ServiceUnavailableError,
            BadGatewayError,
            InternalServerError,
            APIConnectionError,
            APIError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
        ),
    ):
        return True
    # Permanent classes (defensive: several subclass openai.APIStatusError).
    if isinstance(
        exc,
        (
            AuthenticationError,
            PermissionDeniedError,
            BadRequestError,
            NotFoundError,
            UnprocessableEntityError,
            InvalidRequestError,
            ContextWindowExceededError,
            ContentPolicyViolationError,
            UnsupportedParamsError,
            BudgetExceededError,
            GuardrailRaisedException,
        ),
    ):
        return False
    status = _exception_status_code(exc)
    if status is not None:
        return status in _RETRYABLE_STATUS_CODES
    # Unknown openai API errors without a status are transport-ish: bounded retry.
    if isinstance(exc, openai.APIError):
        return True
    return False


class LM:
    """
    Language Model class for interacting with various LLM providers.

    This class provides a unified interface for making requests to different language
    model providers through LiteLLM. It supports caching, rate limiting, usage tracking,
    and batch processing for efficient API usage.

    The class maintains separate physical and virtual usage statistics, where:
    - Physical usage: Actual API calls made (with caching applied)
    - Virtual usage: Total usage if no caching was used

    Attributes:
        model (str): Name of the model to use.
        max_ctx_len (int): Maximum context length in tokens.
        max_tokens (int): Maximum number of tokens to generate.
        rate_limit (int | None): Maximum requests per minute.
        max_batch_size (int): Maximum batch size for concurrent requests.
        tokenizer (Tokenizer | None): Custom tokenizer instance.
        kwargs (dict): Configuration parameters for the LLM API.
        stats (LMStats): Usage statistics tracking.
        physical_usage_limit (UsageLimit): Physical usage limits.
        virtual_usage_limit (UsageLimit): Virtual usage limits.
        cache: Cache instance for storing responses.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_ctx_len: int = 128000,
        max_tokens: int | None = None,
        max_batch_size: int = 64,
        rate_limit: int | None = None,
        tpm_limit: int | None = None,
        tokenizer: Tokenizer | None = None,
        cache: Any = None,
        physical_usage_limit: UsageLimit = UsageLimit(),
        virtual_usage_limit: UsageLimit = UsageLimit(),
        max_retries: int = 0,
        retry_backoff_seconds: float = 2.0,
        send_max_completion_tokens: bool = True,
        **kwargs: dict[str, Any],
    ):
        """
        Initialize the Language Model instance.

        Args:
            model (str): Name of the model to use. Defaults to "gpt-4o-mini".
            temperature (float): Sampling temperature. Defaults to 0.0.
            max_ctx_len (int): Maximum context length in tokens. Defaults to 128000.
            max_tokens (int | None): Maximum number of completion tokens. Defaults to
                512, or 8192 for reasoning models (gpt-5, o-series, ...), whose hidden
                reasoning tokens are spent from this same budget. Set explicitly to
                override; pass reasoning_effort (e.g. "minimal") via kwargs to trade
                reasoning depth for cost on supported models.
            max_batch_size (int): Maximum batch size for concurrent requests. Defaults to 64.
            rate_limit (int | None): Maximum requests per minute. If set, caps max_batch_size and adds delays.
            tokenizer (Tokenizer | None): Custom tokenizer instance. Defaults to None.
            cache: Cache instance to use. Defaults to None.
            physical_usage_limit (UsageLimit): Physical usage limits for the model. Defaults to UsageLimit().
            virtual_usage_limit (UsageLimit): Virtual usage limits for the model. Defaults to UsageLimit().
            max_retries (int): Number of retries for transient provider failures
                after the first physical attempt (defaults to 0 = no retries).
            retry_backoff_seconds (float): Base exponential backoff between retries
                (capped at 30 seconds). Defaults to 2.0.
            send_max_completion_tokens (bool): When False, omit the completion-budget
                parameter (max_completion_tokens/tokens) from the API request entirely
                rather than sending ``max_tokens``. Used to match providers/experiments
                that do not specify an output budget. Defaults to True (unchanged
                historical behaviour).
            **kwargs: Additional keyword arguments passed to the underlying LLM API.
        """
        self.model = model
        self.max_ctx_len = max_ctx_len
        if max_tokens is None:
            max_tokens = DEFAULT_REASONING_MAX_TOKENS if self.is_reasoning_model() else DEFAULT_MAX_TOKENS
        self.max_tokens = max_tokens
        self.rate_limit = rate_limit
        self.tpm_limit = tpm_limit
        self._token_usage_history: deque = deque()  # (timestamp, tokens)

        if rate_limit is not None:
            self._rate_limit_delay: float = 60 / rate_limit
            if max_batch_size is not None:
                self.max_batch_size = min(rate_limit, max_batch_size)
            else:
                self.max_batch_size = rate_limit
        else:
            self.max_batch_size = max_batch_size
        self.tokenizer = tokenizer
        # Use max_completion_tokens - LiteLLM translates to max_tokens for older models automatically
        self.kwargs = dict(temperature=temperature, max_completion_tokens=max_tokens, **kwargs)
        # Opt-in escape hatch: when False, __call__ omits every completion-budget
        # parameter (max_completion_tokens / max_tokens / max_output_tokens) from
        # the API request entirely, instead of transmitting self.max_tokens.
        # Defaults to True so existing LOTUS behaviour is unchanged.
        self.send_max_completion_tokens = send_max_completion_tokens

        self.stats: LMStats = LMStats()
        self.physical_usage_limit = physical_usage_limit
        self.virtual_usage_limit = virtual_usage_limit

        if max_retries < 0:
            raise ValueError("max_retries must be >= 0.")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be >= 0.")
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.retry_max_backoff_seconds = 30.0

        self.cache = cache if cache is not None else CacheFactory.create_default_cache()

    def __call__(
        self,
        messages: list[list[dict[str, str]]],
        show_progress_bar: bool = True,
        progress_bar_desc: str = "Processing uncached messages",
        **kwargs: dict[str, Any],
    ) -> LMOutput:
        all_kwargs = {**self.kwargs, **kwargs}

        # Any-Join-aligned mode: omit the completion-budget parameter entirely
        # (not set to None, which some SDKs would serialize). Applies to every
        # spelling of the budget so no budget key reaches the API request.
        if not self.send_max_completion_tokens:
            for _budget_key in ("max_completion_tokens", "max_tokens", "max_output_tokens"):
                all_kwargs.pop(_budget_key, None)

        # Set top_logprobs if logprobs requested
        if all_kwargs.get("logprobs", False):
            all_kwargs.setdefault("top_logprobs", 10)

        if lotus.settings.enable_cache:
            # Check cache and separate cached and uncached messages. Cached entries
            # are re-validated: an invalid/poisoned entry (e.g. a persisted provider
            # failure) is deleted and treated as a miss instead of being replayed.
            hashed_messages = [self._hash_messages(msg, all_kwargs) for msg in messages]
            cached_responses_raw = [self.cache.get(hash) for hash in hashed_messages]
            cached_responses: list[ModelResponse | None] = []
            for hash, resp in zip(hashed_messages, cached_responses_raw):
                if resp is None:
                    cached_responses.append(None)
                elif isinstance(resp, ModelResponse) and self._classify_response(resp) == "valid":
                    cached_responses.append(resp)
                else:
                    # Invalid cached response: purge it and treat the request as a miss.
                    if resp is not None:
                        self.cache.delete(hash)
                    cached_responses.append(None)
        else:
            hashed_messages = []
            cached_responses = []

        uncached_data = (
            [(msg, hash) for msg, hash, resp in zip(messages, hashed_messages, cached_responses) if resp is None]
            if lotus.settings.enable_cache
            else [(msg, "no-cache") for msg in messages]
        )

        self.stats.cache_hits += len(messages) - len(uncached_data)

        # Process uncached messages in batches. Statistics and cache insertion for
        # fresh responses happen per-item inside the execution path, so a validated
        # success is persisted immediately and a failed response is never cached.
        uncached_responses = self._process_uncached_messages(
            uncached_data, all_kwargs, show_progress_bar, progress_bar_desc
        )

        # Update virtual stats for cached responses
        if lotus.settings.enable_cache:
            for resp in cached_responses:
                if resp is not None and isinstance(resp, ModelResponse):
                    self._update_stats(resp, is_cached=True)

        # Merge all responses in original order and extract outputs
        all_responses = (
            self._merge_responses(cached_responses, uncached_responses)
            if lotus.settings.enable_cache
            else uncached_responses
        )
        outputs: list[str] = [self._get_top_choice(resp) for resp in all_responses]
        logprobs = (
            [self._get_top_choice_logprobs(resp) for resp in all_responses] if all_kwargs.get("logprobs") else None
        )

        return LMOutput(outputs=outputs, logprobs=logprobs)

    def get_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        show_progress_bar: bool = True,
        progress_bar_desc: str = "Processing uncached messages",
        response_format: BaseModel | None = None,
        **kwargs: dict[str, Any],
    ) -> str | BaseModel:
        messages = [
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        ]
        output = self(
            messages,
            show_progress_bar=show_progress_bar,
            progress_bar_desc=progress_bar_desc,
            response_format=response_format,  # type: ignore
            **kwargs,
        ).outputs[0]
        if response_format:
            return response_format.model_validate_json(output)
        return output

    def _process_uncached_messages(
        self,
        uncached_data: list[tuple[list[dict[str, str]], str]],
        all_kwargs: dict[str, Any],
        show_progress_bar: bool,
        progress_bar_desc: str,
    ) -> list[ModelResponse]:
        """
        Process uncached messages in batches and return responses.

        Args:
            uncached_data: List of tuples containing (messages, hash) for uncached messages.
            all_kwargs: Complete keyword arguments for the LLM API.
            show_progress_bar: Whether to show progress bar.
            progress_bar_desc: Description for the progress bar.

        Returns:
            List of ModelResponse objects from the LLM API.
        """
        total_calls = len(uncached_data)

        pbar = tqdm(
            total=total_calls,
            desc=progress_bar_desc,
            disable=not show_progress_bar,
            bar_format="{l_bar}{bar} {n}/{total} LM calls [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

        # (absolute_index, messages, hash) items preserve prompt-to-response
        # correspondence across per-request retries.
        items = [(idx, msg, hash) for idx, (msg, hash) in enumerate(uncached_data)]

        try:
            if self.tpm_limit is not None:
                uncached_responses = self._process_with_tpm_limiting(items, all_kwargs, pbar)
            elif self.rate_limit is not None:
                uncached_responses = self._process_with_rate_limiting(items, all_kwargs, pbar)
            else:
                uncached_responses = self._execute_batch(items, all_kwargs, pbar)
        finally:
            pbar.close()
        return uncached_responses

    def _execute_batch(
        self,
        items: list[tuple[int, list[dict[str, str]], str]],
        all_kwargs: dict[str, Any],
        pbar: tqdm,
    ) -> list[ModelResponse]:
        """Execute one batch of (absolute_index, messages, hash) items.

        Every physical outcome is independently classified:

          VALID      → accounted, persisted to the cache immediately, completed
          RETRYABLE  → any billed usage accounted, NOT cached, retried (bounded)
          FATAL      → any billed usage accounted, NOT cached, raised after the
                       remaining items of the current dispatch have been accounted

        Returns responses in ascending absolute-index order.
        """
        pending = list(items)
        done: dict[int, ModelResponse] = {}
        attempt = 0
        abort_error: BaseException | None = None

        while pending:
            next_pending: list[tuple[int, list[dict[str, str]], str]] = []
            try:
                responses = batch_completion(
                    self.model,
                    [msg for _, msg, _ in pending],
                    drop_params=True,
                    max_workers=min(self.max_batch_size, len(pending)),
                    **all_kwargs,
                )
            except Exception as exc:
                # Whole-batch failure (executor/provider-level raise rather than an
                # in-place result object): retry the whole batch when transient.
                if is_retryable_exception(exc):
                    if attempt >= self.max_retries:
                        raise self._build_retry_exhausted_error(exc, None, attempt + 1) from exc
                    attempt += 1
                    self.stats.retry_attempts += len(pending)
                    self.stats.retryable_failures += len(pending)
                    self._sleep_before_retry(attempt)
                    continue
                self.stats.fatal_failures += 1
                raise

            if len(responses) != len(pending):
                raise RuntimeError(
                    "LiteLLM batch_completion returned an unexpected number of responses: "
                    f"expected {len(pending)}, got {len(responses)}."
                )

            for (abs_idx, msg, hash_key), resp in zip(pending, responses):
                self.stats.physical_requests += 1
                cls = self._classify_response(resp)
                if cls == "valid":
                    self._account_attempt(resp, hash_key, cache=True)
                    done[abs_idx] = resp
                    pbar.update(1)
                elif cls == "retryable":
                    self.stats.retryable_failures += 1
                    self._account_attempt(resp, hash_key, cache=False)
                    if abort_error is not None:
                        continue  # aborting this dispatch; do not schedule more work
                    if attempt >= self.max_retries:
                        abort_error = self._build_retry_exhausted_error(resp, hash_key, attempt + 1)
                        continue
                    next_pending.append((abs_idx, msg, hash_key))
                else:  # fatal
                    self.stats.fatal_failures += 1
                    self._account_attempt(resp, hash_key, cache=False)
                    if abort_error is None:
                        abort_error = self._build_fatal_error(resp)

            if abort_error is not None:
                raise abort_error

            if next_pending:
                attempt += 1
                self.stats.retry_attempts += len(next_pending)
                self._sleep_before_retry(attempt)
                pending = next_pending
            else:
                pending = []

        return [done[i] for i in sorted(done)]

    def _classify_response(self, response: Any) -> str:
        """Classify one batch_completion result.

        Returns one of:
          "valid"     — usable completion, safe to cache and return
          "retryable" — transient provider failure (content=None plus provider
                        error metadata, malformed payload, or a retryable exception)
          "fatal"     — permanent failure (auth/billing/refusal/unknown object)
        """
        if isinstance(response, BaseException):
            return "retryable" if is_retryable_exception(response) else "fatal"

        if not isinstance(response, ModelResponse):
            # Unknown internal/programming object: fail rather than retry blindly.
            return "fatal"

        try:
            message = response.choices[0].message
        except (IndexError, AttributeError):
            # Provider payload without usable choices: transient provider glitch.
            return "retryable"

        if getattr(message, "refusal", None):
            # Content-policy refusal: permanent, do not retry.
            return "fatal"

        content = message.content
        if content is not None:
            # Non-None content (including an empty string) follows native handling:
            # the native pipeline only rejects `content is None`.
            return "valid"

        # content is None: usable only if there is no structural provider error.
        psf = getattr(response, "provider_specific_fields", None)
        if isinstance(psf, dict):
            if psf.get("native_finish_reason") == "error":
                return "retryable"
            error = psf.get("error")
            if isinstance(error, dict):
                code = error.get("code")
                if isinstance(code, int) and code in _RETRYABLE_STATUS_CODES:
                    return "retryable"
                metadata = error.get("metadata")
                error_type = metadata.get("error_type") if isinstance(metadata, dict) else None
                if isinstance(error_type, str) and error_type in _RETRYABLE_PROVIDER_ERROR_TYPES:
                    return "retryable"
        # Ambiguous content=None without refusal or provider error metadata: never
        # cache; a bounded retry is safer than an immediate crash, and with
        # max_retries=0 it raises after the first attempt (like native behavior).
        return "retryable"

    def _response_usage(self, response: Any) -> Any | None:
        """Usage attached to a physical outcome, if any. Exceptions have none."""
        if isinstance(response, BaseException):
            return None
        usage = getattr(response, "usage", None)
        return usage if usage is not None else None

    def _account_attempt(self, response: Any, hash: str, cache: bool) -> None:
        """Account one physical outcome and optionally persist a validated success.

        Billed usage is counted for every response that carries usage, including
        retryable/fatal failures. Only validated successes are cached.
        """
        if self._response_usage(response) is not None:
            self._update_stats(response, is_cached=False)
        if cache and lotus.settings.enable_cache and hash != "no-cache":
            self._cache_response(response, hash)

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = min(
            self.retry_max_backoff_seconds,
            self.retry_backoff_seconds * (2 ** max(0, attempt - 1)),
        )
        time.sleep(delay)

    @staticmethod
    def _describe_outcome(outcome: Any) -> str:
        """Short, sanitized description of a failed outcome (no full prompts/keys)."""
        if isinstance(outcome, BaseException):
            return f"{type(outcome).__name__}: {str(outcome)[:300]}"
        if isinstance(outcome, ModelResponse):
            try:
                content = outcome.choices[0].message.content
                finish_reason = outcome.choices[0].finish_reason
            except Exception:
                content, finish_reason = "<unavailable>", None
            psf = getattr(outcome, "provider_specific_fields", None)
            error = psf.get("error") if isinstance(psf, dict) else None
            return f"ModelResponse(content={content!r}, provider_error={error}, finish_reason={finish_reason})"
        return f"{type(outcome).__name__}: {str(outcome)[:300]}"

    def _build_fatal_error(self, outcome: Any) -> BaseException:
        if isinstance(outcome, BaseException):
            return outcome
        return ValueError(f"Unusable model response: {self._describe_outcome(outcome)}")

    def _build_retry_exhausted_error(
        self, outcome: Any, hash_key: str | None, attempts: int
    ) -> LotusRetryExhaustedError:
        description = self._describe_outcome(outcome)
        return LotusRetryExhaustedError(
            f"Logical model request still failing after {attempts} physical attempt(s): {description}",
            attempts=attempts,
            hash=hash_key,
            model=self.model,
            last_error=description,
        )

    def _process_with_rate_limiting(
        self,
        items: list[tuple[int, list[dict[str, str]], str]],
        all_kwargs: dict[str, Any],
        pbar: tqdm,
    ) -> list[ModelResponse]:
        """
        Process messages with rate limiting applied.

        This method processes messages in batches while respecting the rate limit
        by adding delays between batches to ensure the rate limit is not exceeded.

        Args:
            items: List of (absolute_index, messages, hash) tuples to process.
            all_kwargs: Complete keyword arguments for the LLM API.
            pbar: Progress bar instance to update.

        Returns:
            List of ModelResponse objects from the LLM API.
        """
        responses = []
        num_batches = math.ceil(len(items) / self.max_batch_size)
        # We know rate_limit is not None because we're in the rate limiting branch
        assert self.rate_limit is not None
        min_interval_per_request = 60 / self.rate_limit  # seconds per request

        for i in range(num_batches):
            start_time = time.time()
            start_idx = i * self.max_batch_size
            end_idx = min((i + 1) * self.max_batch_size, len(items))
            sub_items = items[start_idx:end_idx]
            sub_responses = self._execute_batch(sub_items, all_kwargs, pbar)
            responses.extend(sub_responses)
            end_time = time.time()
            elapsed = end_time - start_time

            # Calculate required delay based on number of requests in this batch
            # Each request should be spaced by min_interval_per_request
            required_time_for_batch = len(sub_items) * min_interval_per_request

            # Only sleep if the batch was faster than the required time
            if i < num_batches - 1:  # Don't sleep after the last batch
                to_sleep = required_time_for_batch - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
        return responses

    def _get_tokens_used_in_last_minute(self) -> int:
        current_time = time.time()
        while self._token_usage_history and self._token_usage_history[0][0] < current_time - 60:
            self._token_usage_history.popleft()
        return sum(tokens for _, tokens in self._token_usage_history)

    def _process_with_tpm_limiting(
        self,
        items: list[tuple[int, list[dict[str, str]], str]],
        all_kwargs: dict[str, Any],
        pbar: tqdm,
    ) -> list[ModelResponse]:
        assert self.tpm_limit is not None
        responses = []
        token_estimates = []
        max_allowed_tpm = int(self.tpm_limit * 0.95)

        for idx, item in enumerate(items):
            msg = item[1]
            est = self.count_tokens(msg)
            total_est = est + self.max_tokens

            if total_est > max_allowed_tpm:
                raise ValueError(
                    f"Row {idx} estimated size ({total_est} tokens) exceeds your "
                    f"total TPM limit with safety buffer ({max_allowed_tpm} tokens). "
                    f"This row is too large to ever be sent on your current API tier. "
                    f"Please trim your data or upgrade your account tier."
                )
            token_estimates.append(est)

        i = 0
        while i < len(items):
            tokens_in_last_minute = self._get_tokens_used_in_last_minute()
            # Use a 5% safety buffer to avoid tight boundary errors
            available_tokens = max(0, int(self.tpm_limit * 0.95) - tokens_in_last_minute)

            # Build sub-batch based on available tokens and optional rate (RPM) limit
            sub_items = []
            sub_batch_estimate = 0

            # Determine max possible requests in this specific burst
            # based on self.max_batch_size AND optional self.rate_limit
            effective_max_batch = self.max_batch_size

            while i < len(items):
                # Include max_tokens in estimate as buffer for the completion
                est = token_estimates[i] + self.max_tokens

                if sub_batch_estimate + est <= available_tokens:
                    sub_items.append(items[i])
                    sub_batch_estimate += est
                    i += 1
                    if len(sub_items) >= effective_max_batch:
                        break
                else:
                    # Row doesn't fit in current budget. Stop here.
                    break

            if sub_items:
                start_time = time.time()
                sub_responses = self._execute_batch(sub_items, all_kwargs, pbar)
                responses.extend(sub_responses)

                # Record actual usage from response
                actual_tokens = sum(r.usage.total_tokens for r in sub_responses if hasattr(r, "usage"))
                self._token_usage_history.append((start_time, actual_tokens))

                # IF rate_limit is set, we must also enforce the RPM-based delay
                # to prevent hitting RPM limits with many small requests.
                if self.rate_limit is not None:
                    elapsed = time.time() - start_time
                    required_time_for_rpm = len(sub_items) * (60 / self.rate_limit)
                    to_sleep = required_time_for_rpm - elapsed
                    if to_sleep > 0:
                        time.sleep(to_sleep)
            else:
                # Need to wait for token window to clear
                wait_time = 1.0
                if self._token_usage_history:
                    wait_time = max(0.1, self._token_usage_history[0][0] + 60.1 - time.time())

                pbar.set_postfix_str(f"TPM Limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                pbar.set_postfix_str("")

        return responses

    def _cache_response(self, response: ModelResponse, hash: str) -> None:
        """
        Cache a response and update stats if successful.

        Args:
            response: ModelResponse object to cache.
            hash: Hash key for the cache entry.

        Raises:
            OpenAIError: If the response contains an error.
        """
        if isinstance(response, OpenAIError):
            raise response
        self.cache.insert(hash, response)

    def _hash_messages(self, messages: list[dict[str, str]], kwargs: dict[str, Any]) -> str:
        """Hash messages and kwargs to create a unique key for the cache"""
        to_hash = str(self.model) + str(messages) + str(kwargs)
        return hashlib.sha256(to_hash.encode()).hexdigest()

    def _merge_responses(
        self, cached_responses: list[ModelResponse | None], uncached_responses: list[ModelResponse]
    ) -> list[ModelResponse]:
        """Merge cached and uncached responses, maintaining order"""
        uncached_iter = iter(uncached_responses)
        return [resp if resp is not None else next(uncached_iter) for resp in cached_responses]

    def _check_usage_limit(self, usage: LMStats.TotalUsage, limit: UsageLimit, usage_type: str):
        """Helper to check if usage exceeds limits"""
        if (
            usage.prompt_tokens > limit.prompt_tokens_limit
            or usage.completion_tokens > limit.completion_tokens_limit
            or usage.total_tokens > limit.total_tokens_limit
            or usage.total_cost > limit.total_cost_limit
        ):
            raise LotusUsageLimitException(f"Usage limit exceeded. Current {usage_type} usage: {usage}, Limit: {limit}")

    def _update_usage_stats(self, usage: LMStats.TotalUsage, response: ModelResponse, cost: float | None):
        """Helper to update usage statistics including cached tokens"""
        if hasattr(response, "usage") and response.usage:
            usage.prompt_tokens += response.usage.prompt_tokens or 0
            usage.completion_tokens += response.usage.completion_tokens or 0
            usage.total_tokens += response.usage.total_tokens or 0
            if cost is not None:
                usage.total_cost += cost

            # Extract cached token information from prompt_tokens_details if available
            if hasattr(response.usage, "prompt_tokens_details") and response.usage.prompt_tokens_details:
                details = response.usage.prompt_tokens_details
                if isinstance(details, dict):
                    cached_tokens = details.get("cached_tokens", 0) or 0
                    cache_creation_tokens = details.get("cache_creation_tokens", 0) or 0
                else:
                    cached_tokens = getattr(details, "cached_tokens", 0) or 0
                    cache_creation_tokens = getattr(details, "cache_creation_tokens", 0) or 0

                usage.cached_prompt_tokens += cached_tokens
                usage.cache_creation_tokens += cache_creation_tokens

            # Extract hidden reasoning tokens (o-series / gpt-5) if provided.
            completion_details = getattr(response.usage, "completion_tokens_details", None)
            if completion_details:
                if isinstance(completion_details, dict):
                    reasoning_tokens = completion_details.get("reasoning_tokens", 0) or 0
                else:
                    reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0
                usage.reasoning_tokens += reasoning_tokens

    def _response_cost(self, response: ModelResponse) -> float | None:
        """Provider-billed cost when available, else LiteLLM's price calculation.

        Preference order:
          1. response.usage.cost_details.upstream_inference_cost (OpenRouter's
             own billed figure; present even on partial/failed responses)
          2. litellm.completion_cost via lotus.pricing (model price table)
          3. None (existing no-pricing warning path)
        The `usage.cost` field is deliberately not used: OpenRouter reports 0 there.
        """
        usage = getattr(response, "usage", None)
        if usage is not None:
            cost_details = getattr(usage, "cost_details", None)
            if isinstance(cost_details, dict):
                cost = cost_details.get("upstream_inference_cost")
            else:
                cost = (
                    getattr(cost_details, "upstream_inference_cost", None)
                    if cost_details is not None
                    else None
                )
            if cost is not None:
                try:
                    cost = float(cost)
                except (TypeError, ValueError):
                    cost = None
                if cost is not None and cost > 0:
                    return cost
        return calculate_cost_from_response(response)

    def _update_stats(self, response: ModelResponse, is_cached: bool = False):
        """
        Update usage statistics from a model response.

        Uses LiteLLM's completion_cost() function which:
        - Maintains up-to-date pricing for 100+ models
        - Automatically handles cached tokens and their different pricing
        - Supports all token types (input, cached, cache creation, output, image)

        Args:
            response: The ModelResponse from LiteLLM
            is_cached: Whether this response came from Lotus's operator cache
        """
        if not hasattr(response, "usage"):
            return

        # Use provider cost_details when available, else LiteLLM's completion_cost
        cost = self._response_cost(response)

        if cost is not None:
            lotus.logger.debug(f"Calculated cost: ${cost:.6f} for model {self.model}")
        else:
            lotus.logger.debug(f"No pricing information available for model {self.model}")
            warnings.warn(f"No pricing information available for model {self.model}. ")

        # Always update virtual usage
        self._update_usage_stats(self.stats.virtual_usage, response, cost)
        self._check_usage_limit(self.stats.virtual_usage, self.virtual_usage_limit, "virtual")

        # Only update physical usage for non-cached responses
        if not is_cached:
            self._update_usage_stats(self.stats.physical_usage, response, cost)
            self._check_usage_limit(self.stats.physical_usage, self.physical_usage_limit, "physical")

    def _get_top_choice(self, response: ModelResponse) -> str:
        # Handle authentication errors and other exceptions
        if isinstance(response, (AuthenticationError, OpenAIError)):
            raise response

        choice = response.choices[0]
        assert isinstance(choice, Choices)
        if choice.finish_reason == "length":
            fix_hint = (
                f"Raise the budget when configuring the model, e.g.: "
                f'lotus.settings.configure(lm=LM(model="{self.model}", max_tokens={self.max_tokens * 2}))'
            )
            lotus.logger.warning(
                f"Completion from {self.model} was truncated by the max_tokens limit "
                f"({self.max_tokens}). "
                + (
                    "This is a reasoning model: hidden reasoning tokens are spent from the same "
                    "budget, so an exhausted budget can return an empty answer that operators "
                    f"silently coerce to their default output. {fix_hint}, or lower the "
                    'reasoning depth with LM(..., reasoning_effort="minimal").'
                    if self.is_reasoning_model()
                    else f"{fix_hint}."
                )
            )
        if choice.message.content is None:
            raise ValueError(f"No content in response: {response}")
        return choice.message.content

    def _get_top_choice_logprobs(self, response: ModelResponse) -> list[ChatCompletionTokenLogprob]:
        # Handle authentication errors and other exceptions
        if isinstance(response, (AuthenticationError, OpenAIError)):
            raise response

        choice = response.choices[0]
        assert isinstance(choice, Choices)
        assert choice.logprobs is not None and isinstance(choice.logprobs, ChoiceLogprobs)
        logprobs = choice.logprobs["content"]
        return logprobs

    def format_logprobs_for_cascade(self, logprobs: list[list[ChatCompletionTokenLogprob]]) -> LogprobsForCascade:
        all_tokens = []
        all_confidences = []
        for resp_logprobs in logprobs:
            tokens = [logprob.token for logprob in resp_logprobs]
            confidences = [np.exp(logprob.logprob) for logprob in resp_logprobs]
            all_tokens.append(tokens)
            all_confidences.append(confidences)
        return LogprobsForCascade(tokens=all_tokens, confidences=all_confidences)

    def format_logprobs_for_filter_cascade(
        self,
        logprobs: list[list[ChatCompletionTokenLogprob]],
        output_tokens: tuple[str, str] = ("True", "False"),
    ) -> LogprobsForFilterCascade:
        positive_token, negative_token = output_tokens
        base_cascade = self.format_logprobs_for_cascade(logprobs)
        all_positive_probs = []

        def get_normalized_positive_prob(token_probs: dict[str, float]) -> float | None:
            if positive_token in token_probs and negative_token in token_probs:
                pos_prob = token_probs[positive_token]
                neg_prob = token_probs[negative_token]
                return pos_prob / (pos_prob + neg_prob)
            return None

        for resp_idx, response_logprobs in enumerate(logprobs):
            pos_prob = None
            for logprob in response_logprobs:
                token_probs = {top.token: np.exp(top.logprob) for top in logprob.top_logprobs}
                pos_prob = get_normalized_positive_prob(token_probs)
                if pos_prob is not None:
                    break

            if pos_prob is None:
                pos_prob = 1 if positive_token in base_cascade.tokens[resp_idx] else 0

            all_positive_probs.append(pos_prob)

        return LogprobsForFilterCascade(
            tokens=base_cascade.tokens, confidences=base_cascade.confidences, positive_probs=all_positive_probs
        )

    def count_tokens(self, messages: list[dict[str, str]] | str) -> int:
        """Count tokens in messages using either custom tokenizer or model's default tokenizer"""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        custom_tokenizer: dict[str, Any] | None = None
        if self.tokenizer:
            custom_tokenizer = dict(type="huggingface_tokenizer", tokenizer=self.tokenizer)

        return token_counter(
            custom_tokenizer=custom_tokenizer,
            model=self.model,
            messages=messages,
        )

    def encode_text(self, text: str) -> list[int]:
        """Encode text into tokens using either custom tokenizer or model's default tokenizer"""
        custom_tokenizer: dict[str, Any] | None = None
        if self.tokenizer:
            custom_tokenizer = dict(type="huggingface_tokenizer", tokenizer=self.tokenizer)
        return encode(model=self.model, text=text, custom_tokenizer=custom_tokenizer)

    def decode_tokens(self, tokens: list[int]) -> str:
        """Decode tokens into text using either custom tokenizer or model's default tokenizer"""
        custom_tokenizer: dict[str, Any] | None = None
        if self.tokenizer:
            custom_tokenizer = dict(type="huggingface_tokenizer", tokenizer=self.tokenizer)
        return decode(model=self.model, tokens=tokens, custom_tokenizer=custom_tokenizer)

    def print_total_usage(self):
        print("\n=== Usage Statistics ===")
        print("Virtual  = Total usage if no caching was used")
        print("Physical = Actual usage with caching applied\n")
        print(f"Virtual Cost:     ${self.stats.virtual_usage.total_cost:,.6f}")
        print(f"Physical Cost:    ${self.stats.physical_usage.total_cost:,.6f}")
        print(f"Virtual Tokens:   {self.stats.virtual_usage.total_tokens:,}")
        print(f"Physical Tokens:  {self.stats.physical_usage.total_tokens:,}")
        print(f"Cache Hits:       {self.stats.cache_hits:,}\n")

    def reset_stats(self):
        self.stats = LMStats()

    def reset_cache(self, max_size: int | None = None):
        self.cache.reset(max_size)

    def get_model_name(self) -> str:
        raw_model = self.model
        if not raw_model:
            return ""

        # If a slash is present, assume the model name is after the last slash.
        if "/" in raw_model:
            candidate = raw_model.split("/")[-1]
        else:
            candidate = raw_model

        # If a colon is present, assume the model version is appended and remove it.
        if ":" in candidate:
            candidate = candidate.split(":")[0]

        return candidate.lower()

    def is_deepseek(self) -> bool:
        model_name = self.get_model_name()
        return model_name.startswith("deepseek-r1")

    def is_reasoning_model(self) -> bool:
        """Whether the model spends hidden reasoning tokens from the completion budget
        (e.g. gpt-5 and the o-series)."""
        if _supports_reasoning is None:
            # litellm too old to know — assume non-reasoning (keeps the 512 default).
            return False
        try:
            return _supports_reasoning(model=self.model)
        except Exception:
            # Unknown/unmapped models: assume non-reasoning.
            return False
