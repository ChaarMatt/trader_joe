"""
Rate limiter implementation with token bucket algorithm.
"""

import time
import asyncio
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

class TokenBucket:
    """Token bucket rate limiter implementation."""
    
    def __init__(self, tokens: int, fill_rate: float):
        """
        Initialize the token bucket.
        
        Args:
            tokens: Maximum number of tokens in the bucket
            fill_rate: Rate at which tokens are added (tokens per second)
        """
        self.capacity = tokens
        self.tokens = tokens
        self.fill_rate = fill_rate
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """
        Acquire a token from the bucket.
        
        Returns:
            The time to wait (in seconds) before the token can be used
        """
        async with self._lock:
            now = time.time()
            # Add new tokens based on time elapsed
            elapsed = now - self.last_update
            new_tokens = elapsed * self.fill_rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_update = now

            # Check if we have enough tokens
            if self.tokens >= 1:
                self.tokens -= 1
                return 0.0
            else:
                # Calculate wait time
                wait_time = (1 - self.tokens) / self.fill_rate
                return wait_time

class RateLimiter:
    """Advanced rate limiter with per-endpoint tracking and retry handling."""
    
    def __init__(self):
        self._buckets: Dict[str, TokenBucket] = {}
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._max_retries = 3
        self._base_backoff = 1.0  # Base backoff time in seconds
        
    def add_limit(self, endpoint: str, tokens: int, fill_rate: float):
        """Add rate limit for an endpoint."""
        self._buckets[endpoint] = TokenBucket(tokens, fill_rate)
        
    async def acquire(self, endpoint: str, retry_count: int = 0) -> float:
        """
        Acquire permission to make a request to an endpoint.
        
        Args:
            endpoint: The endpoint to acquire permission for
            retry_count: Current retry attempt number
            
        Returns:
            Wait time in seconds before the request can be made
        """
        if endpoint not in self._buckets:
            return 0.0
            
        wait_time = await self._buckets[endpoint].acquire()
        
        if wait_time > 0 and retry_count < self._max_retries:
            # Add exponential backoff for retries
            backoff = self._base_backoff * (2 ** retry_count)
            return wait_time + backoff
            
        return wait_time
        
    def cache_get(self, key: str, ttl: timedelta) -> Optional[Any]:
        """Get a value from cache if not expired."""
        if key in self._cache:
            value, expiry = self._cache[key]
            if datetime.now() < expiry:
                return value
            del self._cache[key]
        return None
        
    def cache_set(self, key: str, value: Any, ttl: timedelta):
        """Set a value in cache with expiration."""
        self._cache[key] = (value, datetime.now() + ttl)
        
    def cache_clear(self):
        """Clear expired cache entries."""
        now = datetime.now()
        expired = [k for k, (_, exp) in self._cache.items() if exp < now]
        for k in expired:
            del self._cache[k]
            
    async def wait_and_retry(self, endpoint: str, retry_count: int = 0) -> bool:
        """
        Wait for rate limit and handle retries.
        
        Args:
            endpoint: The endpoint being accessed
            retry_count: Current retry attempt number
            
        Returns:
            True if should retry, False if max retries exceeded
        """
        if retry_count >= self._max_retries:
            return False
            
        wait_time = await self.acquire(endpoint, retry_count)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            return True
            
        return True
