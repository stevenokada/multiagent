from mindvirus.board import Board, Post, MODERATOR


def test_feed_windowing_and_order():
    b = Board()
    for i in range(30):
        b.add(Post(round=i, author=f"a{i}", text=f"t{i}"))
    feed = b.feed(25)
    assert len(feed) == 25
    assert feed[0].text == "t5" and feed[-1].text == "t29"
    assert len(b.feed(100)) == 30


def test_render_feed_format():
    b = Board()
    b.add(Post(round=0, author=MODERATOR, text="Welcome"))
    b.add(Post(round=1, author="Maria", text="Hi all"))
    out = b.render_feed(25)
    assert out.splitlines() == ["[round 0] MODERATOR: Welcome", "[round 1] Maria: Hi all"]


def test_posts_since():
    b = Board()
    b.add(Post(0, MODERATOR, "x"))
    idx = len(b.posts)
    b.add(Post(1, "Maria", "y"))
    assert [p.text for p in b.posts_since(idx)] == ["y"]
