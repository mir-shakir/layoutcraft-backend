import time
import functools
from httpx import ReadError
import logging

logger = logging.getLogger(__name__) 
def retry_on_ssLError(func):
    """
    A decorator to retry a function call if it fails with a specific
    SSL DECRYPTION_FAILED_OR_BAD_RECORD_MAC error.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error("NEEDS_ATTENTION")
                # Check if it's the specific intermittent SSL error
                if "DECRYPTION_FAILED_OR_BAD_RECORD_MAC" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"SSL error on {func.__name__}. Retrying in 0.5 seconds... (Attempt {attempt + 1}/{max_retries})")
                    logger.error("NEEDS_ATTENTION")
                    time.sleep(0.5) # Wait for a second before retrying
                else:
                    logger.error(f"Failed on last attempt or due to a different error: {e}")
                    raise e
    return wrapper