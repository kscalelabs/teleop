import asyncio

# A shared variable
shared_counter = 0

async def increment_counter(lock):
    global shared_counter
    async with lock:
        # Critical section starts
        await asyncio.sleep(1)  # Simulate some work
        shared_counter += 1
        # Critical section ends
        print(f"Counter incremented to: {shared_counter}")

async def main():
    lock = asyncio.Lock()
    # Create multiple tasks that modify the shared variable
    tasks = [asyncio.create_task(increment_counter(lock)) for _ in range(5)]
    await asyncio.gather(*tasks)

asyncio.run(main())