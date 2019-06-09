import asyncio
import threading

class Async_Runner():
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._start_async_thread()

    def add_task(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def _start_async_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _start_async_thread(self):
        t = threading.Thread(target=self._start_async_loop)
        t.start()