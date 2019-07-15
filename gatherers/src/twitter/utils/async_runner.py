import asyncio
import threading

# Simple asyncio wrapper for a parallel event loop running in a dedicated thread, this allows for coroutine injections
# without blocking the current thread
class Async_Runner():
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._start_async_thread()

    # Inject a coroutine into this thread, using a threadsafe function
    def add_task(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    # Initialize the nested thread event loop
    def _start_async_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    # Initialize the nested thread and start a loop in it
    def _start_async_thread(self):
        t = threading.Thread(target=self._start_async_loop)
        t.start()