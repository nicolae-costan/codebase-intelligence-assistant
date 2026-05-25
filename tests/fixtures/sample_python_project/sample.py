"""Fixture module for semantic chunker tests."""


def top_level(left: int, right: int) -> int:
    """Add two values."""

    return left + right


async def fetch_user(user_id: str) -> str:
    """Fetch a user identifier asynchronously."""

    return user_id


class Greeter:
    """Build greeting messages."""

    prefix = "Hello"

    def greet(self, name: str) -> str:
        """Build a synchronous greeting."""

        return f"{self.prefix}, {name}"

    async def greet_async(self, name: str) -> str:
        """Build an asynchronous greeting."""

        return self.greet(name)
