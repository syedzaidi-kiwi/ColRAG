import os
import pickle
from functools import wraps
from typing import Any, Callable
from colrag.config import config
from colrag.logger import get_logger

logger = get_logger(__name__)

def cache_result(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cache_dir = config.CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create a unique cache key based on the function name and arguments
        cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
        cache_file = os.path.join(cache_dir, f"{cache_key}.pickle")
        
        if os.path.exists(cache_file):
            logger.info(f"Loading cached result for {func.__name__}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        result = func(*args, **kwargs)
        
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        logger.info(f"Cached result for {func.__name__}")
        return result
    
    return wrapper