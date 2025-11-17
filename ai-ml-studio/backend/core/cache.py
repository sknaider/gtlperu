"""
Redis Cache Manager for AI ML Studio
"""

import redis
import json
from typing import Any, Optional
from .config import settings
from loguru import logger
import pickle


class CacheManager:
    """Redis cache manager"""

    def __init__(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=False,  # We'll handle encoding
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if not self.redis_client or not settings.cache_enabled:
            return None

        try:
            value = self.redis_client.get(key)
            if value is None:
                return None

            # Try to unpickle first (for complex objects)
            try:
                return pickle.loads(value)
            except:
                # If unpickle fails, try JSON
                try:
                    return json.loads(value.decode())
                except:
                    # If JSON fails, return as string
                    return value.decode()

        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if not provided)

        Returns:
            True if successful
        """
        if not self.redis_client or not settings.cache_enabled:
            return False

        try:
            ttl = ttl or settings.cache_ttl

            # Try to pickle complex objects
            try:
                serialized = pickle.dumps(value)
            except:
                # If pickle fails, try JSON
                try:
                    serialized = json.dumps(value).encode()
                except:
                    # If JSON fails, convert to string
                    serialized = str(value).encode()

            self.redis_client.setex(key, ttl, serialized)
            return True

        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete key from cache

        Args:
            key: Cache key

        Returns:
            True if successful
        """
        if not self.redis_client:
            return False

        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False

    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        if not self.redis_client:
            return False

        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Error checking cache existence: {e}")
            return False

    def clear(self) -> bool:
        """
        Clear all cache (use with caution!)

        Returns:
            True if successful
        """
        if not self.redis_client:
            return False

        try:
            self.redis_client.flushdb()
            logger.warning("Cache cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def get_stats(self) -> dict:
        """
        Get cache statistics

        Returns:
            Cache stats dictionary
        """
        if not self.redis_client:
            return {}

        try:
            info = self.redis_client.info()
            return {
                "used_memory": info.get("used_memory_human"),
                "total_keys": self.redis_client.dbsize(),
                "connected_clients": info.get("connected_clients"),
                "uptime_days": info.get("uptime_in_days"),
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}

    def cache_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from arguments

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        return ":".join(key_parts)


# Global cache instance
cache = CacheManager()


def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    """
    Decorator for caching function results

    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key

    Returns:
        Decorated function
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:"
            cache_key += cache.cache_key(*args, **kwargs)

            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return result

            # Execute function
            logger.debug(f"Cache miss: {cache_key}")
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator
