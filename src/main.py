from src.chat.context import ChatContext
import asyncio


async def main():
    chat = ChatContext.get_instance()
    await chat.run()


if __name__ == '__main__':
    asyncio.run(main())
