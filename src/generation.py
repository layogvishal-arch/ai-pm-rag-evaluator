"""
generation.py — Answer generation via the Claude API.

Responsibility: take a user question and a list of retrieved context passages,
build a prompt, call Claude, and return the generated answer as a string.

Design choices:
  - Streaming is used (via .stream() + .get_final_message()) so that long
    responses don't time out on slow networks and token usage is still
    accessible after the stream completes.
  - The context passages are injected inside a clearly delimited <context>
    block so Claude can distinguish retrieved facts from the question itself.
  - The system prompt instructs Claude to ground its answer in the supplied
    context and to say "I don't know" rather than hallucinate when the context
    is insufficient.
"""

from typing import List

import anthropic

from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
# Keeping the system prompt stable (no dynamic content) makes it a good
# candidate for prompt caching on repeated calls.
SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions based solely on the \
provided context passages.

Rules:
1. Only use information from the <context> block to answer.
2. If the context does not contain enough information, say \
"I don't have enough information in the provided documents to answer that."
3. Cite the source document when possible (use the metadata in the context).
4. Be concise and precise.
"""


def generate_answer(
    question: str,
    context_chunks: List[str],
    model: str = CLAUDE_MODEL,
    max_tokens: int = 2048,
) -> str:
    """Call Claude with the retrieved context and return its answer.

    The function constructs a user message that contains:
      - A <context> block with the retrieved passages numbered for readability.
      - The user's original question.

    Streaming is used so that:
      a) HTTP request timeouts are not triggered by slow generation.
      b) The caller could iterate over the stream for real-time display
         (not done here, but easy to add by exposing the stream object).
    .get_final_message() blocks until the stream finishes and returns the
    complete Message object including token usage.

    Args:
        question:       The user's natural-language question.
        context_chunks: Ordered list of retrieved text passages (most relevant
                        first) from retrieval.search().
        model:          Claude model ID.  Defaults to CLAUDE_MODEL in config.py.
        max_tokens:     Upper bound on the length of Claude's reply.

    Returns:
        Claude's answer as a plain string.

    Raises:
        anthropic.AuthenticationError: if ANTHROPIC_API_KEY is invalid.
        anthropic.APIConnectionError:  on network failures.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Format the retrieved passages into a readable numbered list.
    context_block = "\n\n".join(
        f"[{i + 1}] {chunk}" for i, chunk in enumerate(context_chunks)
    )

    user_message = f"""\
<context>
{context_block}
</context>

Question: {question}"""

    # Use streaming so long answers don't time out.
    # get_final_message() waits for the stream to finish and returns the full
    # Message object — convenient when we only need the final text.
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        final_message = stream.get_final_message()

    # Extract the text from the first (and only) text block in the response.
    answer = next(
        block.text for block in final_message.content if block.type == "text"
    )

    print(f"[generation] Tokens used — input: {final_message.usage.input_tokens}, "
          f"output: {final_message.usage.output_tokens}")
    return answer
