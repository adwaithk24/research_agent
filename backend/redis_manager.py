import redis.asyncio as redis
import json
import logging
from typing import Optional, Dict, Any, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Initialize Redis client
redis_client = redis.Redis(host="redis", port=6379, decode_responses=False)

# Constants
DEFAULT_TIMEOUT = 30  # seconds
PDF_CONTENT_PREFIX = "pdf:content:"
PDF_METADATA_PREFIX = "pdf:metadata:"


async def send_to_redis_stream(
    stream_name: str, message: Dict[str, Union[bytes, Any]], maxlen: int = 1000
) -> str:
    """Send a message to a Redis Stream asynchronously.

    Args:
        stream_name: Name of the Redis stream
        message: Dictionary containing the message data. Non-bytes values will be automatically
                JSON serialized to strings. Bytes values will be preserved as-is.
        maxlen: Maximum number of messages to keep in the stream

    Returns:
        str: Message ID from Redis
    """
    try:
        # Process message values - serialize all non-bytes values to JSON
        processed_message = {}
        for k, v in message.items():
            if isinstance(v, bytes):
                processed_message[k] = v
            else:
                # Serialize all non-bytes values to JSON strings
                try:
                    processed_message[k] = json.dumps(v)
                except (TypeError, ValueError):
                    # Handle non-JSON serializable objects
                    processed_message[k] = str(v)

        logger.info("Sent message to stream")
        return await redis_client.xadd(
            stream_name, processed_message, maxlen=maxlen, approximate=True
        )
    except redis.RedisError as e:
        logger.error(f"Error sending message to stream {stream_name}: {e}")
        raise


async def receive_from_redis_stream(
    stream_name: str, last_id: str = "0", timeout: int = DEFAULT_TIMEOUT
) -> list:
    """Receive a message from a Redis Stream with timeout asynchronously.

    Args:
        stream_name: Name of the Redis stream
        last_id: Last message ID received
        timeout: Time to wait for new messages in seconds

    Returns:
        list: List of messages from the stream
    """
    try:
        logger.info("Receiving message from stream")
        return await redis_client.xread(
            {stream_name: last_id}, count=1, block=timeout * 1000
        )
    except redis.RedisError as e:
        logger.error(f"Error receiving message from stream {stream_name}: {e}")
        raise


async def store_pdf_content(
    pdf_id: str, content: Union[str, bytes], metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Store PDF content and metadata in Redis.

    Args:
        pdf_id: Unique identifier for the PDF
        content: PDF content as string or bytes
        metadata: Optional metadata dictionary

    Returns:
        bool: True if storage was successful
    """
    if not pdf_id:
        logger.error("PDF ID is required")
        return False

    if not content:
        logger.error("PDF content is required")
        return False

    try:
        # Ensure content is in bytes
        content_bytes = (
            content if isinstance(content, bytes) else content.encode("utf-8")
        )

        # Store PDF content
        await redis_client.set(
            f"{PDF_CONTENT_PREFIX}{pdf_id}",
            content_bytes,
            ex=86400,  # expire after 24 hours
        )
        logger.info(f"Successfully stored PDF content for ID: {pdf_id}")

        # Store metadata if provided
        if metadata:
            try:
                metadata_json = json.dumps(metadata)
                await redis_client.set(
                    f"{PDF_METADATA_PREFIX}{pdf_id}", metadata_json, ex=86400
                )
                logger.info(f"Successfully stored metadata for PDF ID: {pdf_id}")
            except (TypeError, json.JSONDecodeError) as e:
                logger.error(f"Failed to serialize metadata for PDF {pdf_id}: {e}")
                return False
        return True
    except redis.RedisError as e:
        logger.error(f"Redis error while storing PDF {pdf_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while storing PDF {pdf_id}: {e}")
        return False


async def get_pdf_content(pdf_id: str) -> Optional[bytes]:
    """Retrieve PDF content from Redis.

    Args:
        pdf_id: Unique identifier for the PDF

    Returns:
        Optional[bytes]: PDF content if found, None otherwise
    """
    try:
        return await redis_client.get(f"{PDF_CONTENT_PREFIX}{pdf_id}")
    except redis.RedisError as e:
        logger.error(f"Error retrieving PDF {pdf_id}: {e}")
        return None


async def send_llm_request(
    prompt: str, model_name: str, max_tokens: Optional[int] = None
) -> str:
    """Send LLM request to Redis Stream.

    Args:
        prompt: The prompt to send to the LLM
        model_name: Name of the LLM model to use
        max_tokens: Optional maximum number of tokens for the response

    Returns:
        str: Message ID from Redis
    """
    message = {
        "prompt": prompt,
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
    }
    if max_tokens is not None:
        message["max_tokens"] = max_tokens

    return await send_to_redis_stream("llm_requests", message)


async def receive_llm_response(
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    """Receive LLM response from Redis Stream with timeout.

    Args:
        timeout: Time to wait for response in seconds

    Returns:
        Optional[Dict[str, Any]]: Response dictionary if received, None if timeout
    """
    try:
        # Use '$' as last_id to only get new messages
        response = await receive_from_redis_stream(
            "llm_responses", last_id="$", timeout=timeout
        )
        if response:
            logger.info("Received response from Redis")
            _, messages = response[0]
            # Get the message data and deserialize from JSON
            message_data = messages[0][1]
            deserialized_message = {}
            for k, v in message_data.items():
                key = k.decode() if isinstance(k, bytes) else k
                value = v.decode() if isinstance(v, bytes) else v
                try:
                    # First attempt to parse as JSON
                    parsed = json.loads(value)
                    # If parsed is still a string, try parsing again
                    if isinstance(parsed, str):
                        try:
                            deserialized_message[key] = json.loads(parsed)
                        except json.JSONDecodeError:
                            deserialized_message[key] = parsed
                    else:
                        deserialized_message[key] = parsed
                except json.JSONDecodeError:
                    deserialized_message[key] = value
            return deserialized_message
        else:
            logger.info("No response received from Redis")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding LLM response: {e}")
        return None
