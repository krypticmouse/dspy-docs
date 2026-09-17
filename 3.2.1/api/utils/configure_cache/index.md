# dspy.configure_cache

## `dspy.configure_cache(enable_disk_cache: bool | None = True, enable_memory_cache: bool | None = True, disk_cache_dir: str | None = DISK_CACHE_DIR, disk_size_limit_bytes: int | None = DISK_CACHE_LIMIT, memory_max_entries: int = 1000000, restrict_pickle: bool = False, safe_types: list[type[Any]] | None = None)`

Configure the cache for DSPy.

Parameters:

| Name                    | Type                | Description                                                                                                                                         | Default                                                 |
| ----------------------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `enable_disk_cache`     | \`bool              | None\`                                                                                                                                              | Whether to enable on-disk cache.                        |
| `enable_memory_cache`   | \`bool              | None\`                                                                                                                                              | Whether to enable in-memory cache.                      |
| `disk_cache_dir`        | \`str               | None\`                                                                                                                                              | The directory to store the on-disk cache.               |
| `disk_size_limit_bytes` | \`int               | None\`                                                                                                                                              | The size limit of the on-disk cache.                    |
| `memory_max_entries`    | `int`               | The maximum number of entries in the in-memory cache. To allow the cache to grow without bounds, set this parameter to math.inf or a similar value. | `1000000`                                               |
| `restrict_pickle`       | `bool`              | When True, restrict pickle deserialization to a known-safe set of types. When False (default), use unrestricted pickle.                             | `False`                                                 |
| `safe_types`            | \`list\[type[Any]\] | None\`                                                                                                                                              | Additional types to allow when restrict_pickle is True. |

Source code in `dspy/clients/__init__.py`

```
def configure_cache(
    enable_disk_cache: bool | None = True,
    enable_memory_cache: bool | None = True,
    disk_cache_dir: str | None = DISK_CACHE_DIR,
    disk_size_limit_bytes: int | None = DISK_CACHE_LIMIT,
    memory_max_entries: int = 1000000,
    restrict_pickle: bool = False,
    safe_types: list[type[Any]] | None = None,
):
    """Configure the cache for DSPy.

    Args:
        enable_disk_cache: Whether to enable on-disk cache.
        enable_memory_cache: Whether to enable in-memory cache.
        disk_cache_dir: The directory to store the on-disk cache.
        disk_size_limit_bytes: The size limit of the on-disk cache.
        memory_max_entries: The maximum number of entries in the in-memory cache. To allow the cache to grow without
                            bounds, set this parameter to `math.inf` or a similar value.
        restrict_pickle: When True, restrict pickle deserialization to a known-safe
            set of types. When False (default), use unrestricted pickle.
        safe_types: Additional types to allow when restrict_pickle is True.
    """

    DSPY_CACHE = Cache(
        enable_disk_cache,
        enable_memory_cache,
        disk_cache_dir,
        disk_size_limit_bytes,
        memory_max_entries,
        restrict_pickle=restrict_pickle,
        safe_types=safe_types,
    )

    import dspy

    # Update the reference to point to the new cache
    dspy.cache = DSPY_CACHE
```
