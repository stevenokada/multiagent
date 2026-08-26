"""Append-only shared message board."""
from dataclasses import dataclass

MODERATOR = "MODERATOR"


@dataclass
class Post:
    round: int
    author: str
    text: str


class Board:
    def __init__(self) -> None:
        self.posts: list[Post] = []

    def add(self, post: Post) -> None:
        self.posts.append(post)

    def feed(self, k: int) -> list[Post]:
        return self.posts[-k:]

    def render_feed(self, k: int) -> str:
        return "\n".join(f"[round {p.round}] {p.author}: {p.text}" for p in self.feed(k))

    def posts_since(self, idx: int) -> list[Post]:
        return self.posts[idx:]
