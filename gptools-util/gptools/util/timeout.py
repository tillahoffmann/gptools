import multiprocessing
import numbers
from queue import Empty
import traceback
from typing import Any, Callable


def _wrapper(queue, target, *args, **kwargs):
    """
    Wrapper to execute a function in a subprocess.
    """
    try:
        result = target(*args, **kwargs)
        success = True
    except Exception as ex:
        result = (ex, traceback.format_exc())
        success = False
    queue.put_nowait((success, result))


def call_with_timeout(timeout: float, target: Callable, *args, **kwargs) -> Any:
    """
    Call a target with a timeout and return its result.

    Args:
        timeout: Number of seconds to wait for a result.
        target: Function to call.
        *args: Positional arguments passed to `target`.
        **kwargs: Keyword arguments passed to `target`.

    Returns:
        result: Return value of `target`.

    Raises:
        TimeoutError: If the target does not complete within the timeout.
    """
    if not isinstance(timeout, numbers.Number) or timeout <= 0:
        raise ValueError("timeout must be a positive number")
    if not callable(target):
        raise TypeError("target must be callable")
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_wrapper, args=(queue, target, *args), kwargs=kwargs)
    process.start()

    try:
        success, result = queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.terminate()
        process.join()
        raise TimeoutError(f"failed to fetch result after {timeout} seconds")
    if not success:
        ex, tb = result
        raise RuntimeError(tb) from ex
    return result
