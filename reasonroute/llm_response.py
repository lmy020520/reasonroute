"""Validate billable LLM replies without inventing zero-token usage."""


class LLMResponseError(RuntimeError):
    """A successful HTTP reply cannot be used as a complete generation record."""


class TokenUsageError(LLMResponseError):
    """The provider did not supply valid token accounting for this reply."""


def read_llm_reply(response):
    try:
        data = response.json()
    except ValueError as exc:
        raise LLMResponseError('LLM reply is not valid JSON; token accounting unavailable, sample not recorded') from exc
    return parse_llm_reply(data)


def parse_llm_reply(data):
    usage = data.get('usage') if isinstance(data, dict) else None
    tokens = usage.get('total_tokens') if isinstance(usage, dict) else None
    if type(tokens) is not int or tokens <= 0:
        raise TokenUsageError('LLM reply is missing a positive integer usage.total_tokens; sample not recorded')
    try:
        content = data['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMResponseError('LLM reply has no usable message content; sample not recorded') from exc
    if not isinstance(content, str):
        raise LLMResponseError('LLM message content must be a string; sample not recorded')
    return content, tokens
