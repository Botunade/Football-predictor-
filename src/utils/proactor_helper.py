"""
Windows Proactor Event Loop & Subprocess Compatibility Helper
Ensures Playwright and asyncio subprocess calls execute seamlessly on Windows,
even when Uvicorn is running under SelectorEventLoop (such as with --reload).
"""

import asyncio
import inspect
import logging
import sys
import threading
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


def patch_windows_asyncio():
    """
    Silence Windows asyncio pipe deallocator warnings that occur during
    transport cleanup when processes or pipes close abruptly.
    """
    if sys.platform == "win32":
        try:
            import asyncio.proactor_events
            import asyncio.base_subprocess

            def _silence_del(self):
                try:
                    if hasattr(self, "_sock") and self._sock is not None:
                        try:
                            self._sock.close()
                        except Exception:
                            pass
                except Exception:
                    pass

            asyncio.proactor_events._ProactorBasePipeTransport.__del__ = _silence_del
            asyncio.base_subprocess.BaseSubprocessTransport.__del__ = _silence_del
        except Exception:
            pass


def is_proactor_supported(loop: Optional[asyncio.AbstractEventLoop] = None) -> bool:
    """
    Check if the given or current event loop supports subprocess transports on Windows.
    Returns True on non-Windows platforms.
    """
    if sys.platform != "win32":
        return True

    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return True

    # Windows ProactorEventLoop implements _make_subprocess_transport; SelectorEventLoop does not.
    proactor_cls = getattr(asyncio.windows_events, "ProactorEventLoop", None)
    if proactor_cls is not None and isinstance(loop, proactor_cls):
        return True

    return False


def run_in_proactor_thread(coro_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Execute an async callable or coroutine in a dedicated thread equipped with
    a genuine Windows ProactorEventLoop.
    """
    result = None
    exception = None

    def _worker():
        nonlocal result, exception
        patch_windows_asyncio()
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
        try:
            if inspect.iscoroutinefunction(coro_fn):
                coro = coro_fn(*args, **kwargs)
                result = loop.run_until_complete(coro)
            elif callable(coro_fn):
                res = coro_fn(*args, **kwargs)
                if inspect.iscoroutine(res):
                    result = loop.run_until_complete(res)
                else:
                    result = res
            elif inspect.iscoroutine(coro_fn):
                result = loop.run_until_complete(coro_fn)
            else:
                result = coro_fn
        except Exception as e:
            exception = e
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass

    thread = threading.Thread(target=_worker, name="ProactorWorkerThread", daemon=True)
    thread.start()
    thread.join()

    if exception is not None:
        raise exception

    return result


async def run_coro_with_proactor(coro_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Dispatch a coroutine or async callable safely.
    If running on Windows under a loop without subprocess support (e.g. SelectorEventLoop),
    offloads execution to a dedicated Proactor thread.
    Otherwise awaits directly.
    """
    if sys.platform == "win32" and not is_proactor_supported():
        logger.debug("Active loop lacks Windows subprocess support; offloading to Proactor thread.")
        return await asyncio.to_thread(run_in_proactor_thread, coro_fn, *args, **kwargs)

    if inspect.iscoroutinefunction(coro_fn):
        return await coro_fn(*args, **kwargs)
    elif callable(coro_fn):
        res = coro_fn(*args, **kwargs)
        if inspect.iscoroutine(res):
            return await res
        return res
    elif inspect.iscoroutine(coro_fn):
        return await coro_fn
    return coro_fn
