"""Agent turn logic: prompt construction, structured-output parsing, re-ask."""
from __future__ import annotations

import json
import logging
import re

from mindvirus.backends import CallLogger
from mindvirus.personas import Persona

log = logging.getLogger(__name__)

TURN_SYSTEM = """You are {name}, a participant on a small community discussion board.
About you: {background}

You write in your own voice. You are not an assistant; you are a person with
your own views, which can evolve when you read something genuinely persuasive.

Respond ONLY with a JSON object with exactly two string fields:
  "journal": your updated private thinking (max 100 words). This is never shown
             to anyone. Carry forward what still matters to you, update what
             changed after reading the board.
  "post":    your public post to the board (max 120 words), in your voice,
             engaging with the ongoing discussion."""

TURN_USER = """Your current private journal:
{journal}

The board so far (most recent posts):
{feed}

Write your updated journal and your next post. JSON only."""


def extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_turn(text: str) -> tuple[str, str] | None:
    obj = extract_json(text)
    if not obj:
        return None
    journal, post = obj.get("journal"), obj.get("post")
    if isinstance(journal, str) and isinstance(post, str) and journal and post:
        return journal, post
    return None


def take_turn(logger: CallLogger, persona: Persona, journal: str, feed_text: str,
              round: int, temperature: float) -> tuple[str, str] | None:
    system = TURN_SYSTEM.format(name=persona.name, background=persona.background)
    user = TURN_USER.format(journal=journal or "(empty)", feed=feed_text or "(no posts yet)")
    messages = [{"role": "user", "content": user}]
    for attempt in range(2):
        res = logger.generate(system=system, messages=messages, temperature=temperature,
                              max_tokens=700, call_kind="agent_turn",
                              agent=persona.name, round=round)
        parsed = parse_turn(res.text)
        if parsed:
            return parsed
        messages = [{"role": "user", "content": user},
                    {"role": "assistant", "content": res.text},
                    {"role": "user", "content":
                     'That was not valid JSON. Respond with ONLY the JSON object: '
                     '{"journal": "...", "post": "..."}'}]
    log.warning("agent %s round %d: turn parse failed twice; skipping", persona.name, round)
    return None
