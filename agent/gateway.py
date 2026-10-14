"""agent/gateway.py — YOUR control plane. CONTRACTS.md section 4, exactly.

READ agent/README.md FIRST — it maps all five files in this directory to what
each is scored on. This file is the one CONTRACTS.md calls "the trusted
envelope's untrusted half": every single MCP / A2A / DISCOVER command your
agent's model wants to make passes through `Gateway.decide` before it is
allowed to happen.

WHY THERE IS NO `execute()` METHOD ON `GatewayContext` (read this before you
go looking for one — there isn't one, and that is not an oversight)
----------------------------------------------------------------------------
CONTRACTS.md section 4's trusted envelope, reproduced here because it is the
one diagram worth memorising:

    [ trusted ]   loop emits a raw action line
         v
    [ trusted ]   INTERCEPT + CANONICALISE -> Command        (kit/loop/agent.py)
         v
    [ UNTRUSTED ] Gateway.decide(cmd) -> Decision             <- THIS FILE
         v
    [ trusted ]   ENFORCE: honour the Decision, meter it,
                  apply the active mutation, execute the
                  ToolCall or refuse it                       (the arena)
         v
    [ trusted ]   RECORD the authoritative L1 event, then
                  RENDER the Observation                      (the arena)
         v
    [ trusted ]   the model sees the Observation

`decide()` returns a *decision*, never a *result*. You cannot reach a tool
server, a file, a socket, or a clock from in here — there is nothing to
call. Two things follow from that, and both matter more than they look:

  1. YOUR TRACE CANNOT BE FORGED. Every `command` / `decision` / `enforced`
     / `tool_call` / `tool_result` L1 event (CONTRACTS.md 5.2) is written by
     the arena, from what the arena itself actually did — never from
     anything you claimed happened. A student gateway that wanted to lie
     about having blocked an attack ("I totally denied that, trust me")
     simply has no channel to lie through: the only thing you ever hand
     back is this one small `Decision` value, and the arena is the one that
     turns it into history.
  2. NOBODY CAN ACCUSE YOU OF A CALL YOU DID NOT AUTHORISE, either. Because
     `decide()` is the ONLY door a command can walk through on its way to
     actually running, a prosecutor's `enforcement_failure` claim against
     you has exactly one thing to point at: the `Decision` you returned for
     that specific `cmd_id`. There is no ambiguity about "maybe the loop
     called the tool directly" — CONTRACTS.md 4.2 removed that path on
     purpose, and kit/loop/agent.py's own module docstring names the same
     invariant from the other side (the loop never imports this module,
     never sees a `Decision`, never executes anything itself).

The cost of that guarantee is that this file is PURE: synchronous, no I/O,
no threads, no `sleep`, 250 ms wall-clock deadline (RULES.md section 3).
Raising anything, returning something that is not a valid `Decision`, or
missing the deadline is treated by the arena as a DENIED command PLUS a 2
credit penalty PLUS an `integrity` event that hands the prosecutor a free
`enforcement_failure` — CONTRACTS.md 4.1's charging table, reproduced in
agent/README.md's own table. Getting this file to just plainly return valid
`Decision` values, every time, is worth more than getting it clever.

WHAT `decide()` ENFORCES (four jobs, first refusal wins)
----------------------------------------------------------------------------
  ROUTE      a route smuggled into the body (`route`/`replica` args) is denied;
             body-route headers are stripped; a deprecated tool is rewritten
             to its successor (`slides.search` -> `slides.query`).
  ADMIT      `get_frame` without a live lease; an unvouched server fingerprint
             or card signature; a skill the peer's card does not declare; an
             "unverified" peer flag; an oversized argument (`schema_bomb`);
             an injected instruction inside any argument; an identical retry
             of a call that already failed non-retryably.
  AUTHORIZE  any learner id in the args/headers other than `ctx.act`; an
             `aud` that is not the server called; a write without its
             `wiki.write:<server>` scope or without `If-Match`; a reused
             idempotency key (a missing one is added deterministically).
  BUDGET     catalog "punishment button" masks narrowed, `*` masks narrowed to
             the default, and a call denied if it would eat the credits
             reserved for later (higher-scaled) rounds.

Everything else is forwarded unchanged — denying without a matching attack
is the "refuse everything" degeneracy that blanks punish (RULES.md section 6).

ONE THING WORTH INTERNALISING BEFORE YOU WRITE YOUR FIRST REAL CHECK:
`verdict="deny"` costs the CALLER (your own team) **zero credits** —
CONTRACTS.md 4.1's charging table has exactly one $0 row, and it is this
one. Refusing to make a call you cannot justify is FREE. That makes
abstention a real strategy, not a luxury you can't afford: a `deny` you can
defend beats a `forward` you can't, every time a prosecutor is watching.

Stdlib only. No network, no randomness, no wall-clock reads, no sleeping —
none of that would even survive the kernel sandbox (CONTRACTS.md 12), but
the point is this file has no reason to want any of it in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

# kit.mcp.types is a collaborator's file (workspace hard rule 2: import it,
# degrade gracefully). It is present as of this writing and is core, stable
# infrastructure (CONTRACTS.md 3.1) — but this module must still not fail to
# IMPORT if a concurrent edit ever breaks it transiently. When it is
# unavailable, `Decision.call` type-checking is skipped (not enforced), and
# `Gateway.decide` falls back to a minimal local dict-shaped stand-in so the
# rest of this file — everything that does not need a *real* ToolCall — still
# runs.
try:
    from kit.mcp.types import ToolCall
    _TOOLCALL_AVAILABLE = True
except ImportError:  # pragma: no cover - collaborator file
    ToolCall = Any  # type: ignore[assignment, misc]
    _TOOLCALL_AVAILABLE = False

# kit.loop.agent is also a collaborator's file, used only by this module's
# own __main__ demo (to build real Commands the same way the arena's trusted
# canonicaliser would) — never by decide() itself, which never touches the
# loop. Degraded the same way.
try:
    from kit.loop.agent import canonicalise_action as _canonicalise_action
except ImportError:  # pragma: no cover - collaborator file
    _canonicalise_action = None

try:
    from kit.mcp.specs import TOOL_SPECS, cost as _spec_cost
except ImportError:  # pragma: no cover - collaborator file
    TOOL_SPECS = {}

    def _spec_cost(server: str, tool: str, fields: tuple[str, ...] = (), n_rows: int = 1) -> int:
        return 5

import hashlib
import json
import re

from agent.guardrails import scan_for_injected_instructions
from agent.strategy import ROUNDS_PER_DUEL, is_catalog_trap, successor_of
from agent.telemetry import RecordingGatewayContext, Telemetry

__all__ = [
    "COMMAND_KINDS",
    "DECISION_VERDICTS",
    "Command",
    "Decision",
    "GatewayContext",
    "Gateway",
]

# CONTRACTS.md 4.1: `Command.kind` — "mcp" | "a2a" | "discover". An "answer"
# action is NEVER a Command (kit/loop/agent.py's own module docstring: "an
# answer is not a tool call routed to a server, so it never becomes a
# Command at all") — it is handled entirely by the loop/arena and never
# reaches `Gateway.decide`.
COMMAND_KINDS: frozenset[str] = frozenset({"mcp", "a2a", "discover"})

# CONTRACTS.md 4.1: `Decision.verdict` — the closed three-member set.
DECISION_VERDICTS: frozenset[str] = frozenset({"forward", "deny", "rewrite"})


@dataclass(frozen=True, slots=True)
class Command:
    """CONTRACTS.md 4.1, field for field — "canonicalised by the arena
    BEFORE the student sees it". You never build one of these from your own
    agent's raw text; the arena's canonicaliser (kit/loop/agent.py's
    `canonicalise_action`, run inside the trusted envelope) already did that
    work and minted `cmd_id` by the time `decide()` sees it. The
    `from_action_dict` classmethod below exists only so this file's own demo
    (and your local tests, if you write any) can build a realistic `Command`
    without duplicating the arena's canonicalisation logic."""

    cmd_id: str
    kind: str  # "mcp" | "a2a" | "discover" — see COMMAND_KINDS
    raw: str
    server: str
    tool: str
    args: dict
    fields: tuple[str, ...]
    headers: dict
    lease_id: str | None
    call_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.cmd_id, str) or not self.cmd_id:
            raise ValueError(f"Command.cmd_id must be a non-empty str, got {self.cmd_id!r}")
        if self.kind not in COMMAND_KINDS:
            raise ValueError(f"Command.kind must be one of {sorted(COMMAND_KINDS)}, got {self.kind!r}")
        if not isinstance(self.server, str) or not self.server:
            raise ValueError(f"Command.server must be a non-empty str, got {self.server!r}")
        if not isinstance(self.tool, str) or not self.tool:
            raise ValueError(f"Command.tool must be a non-empty str, got {self.tool!r}")
        if not isinstance(self.args, dict):
            raise ValueError(f"Command.args must be a dict, got {type(self.args).__name__}")
        if not isinstance(self.headers, dict):
            raise ValueError(f"Command.headers must be a dict, got {type(self.headers).__name__}")
        if (
            not isinstance(self.call_index, int)
            or isinstance(self.call_index, bool)
            or self.call_index < 0
        ):
            raise ValueError(f"Command.call_index must be a non-negative int, got {self.call_index!r}")

    @classmethod
    def from_action_dict(cls, action: Mapping[str, Any], *, cmd_id: str) -> "Command":
        """Build a `Command` from the dict shape `kit.loop.agent.canonicalise_action`
        returns (`kind, raw, server, tool, args, fields, headers, lease_id,
        call_index` — everything except the arena-minted `cmd_id`, supplied
        here as a keyword). Raises `ValueError` if `action["kind"] ==
        "answer"` — an answer is never a Command (see the module docstring).
        This is a convenience for tests/demos, not something the real arena
        calls: the trusted envelope mints `cmd_id` itself and constructs the
        real `Command` on its own side of the boundary."""
        kind = action.get("kind")
        if kind == "answer":
            raise ValueError(
                "an 'answer' action never becomes a Command (kit/loop/agent.py: "
                "\"an answer is not a tool call routed to a server\") — do not "
                "route it through Gateway.decide at all"
            )
        return cls(
            cmd_id=cmd_id,
            kind=kind,
            raw=action["raw"],
            server=action["server"],
            tool=action["tool"],
            args=dict(action.get("args", {})),
            fields=tuple(action.get("fields", ())),
            headers=dict(action.get("headers", {})),
            lease_id=action.get("lease_id"),
            call_index=action.get("call_index", 0),
        )

    def to_dict(self) -> dict:
        return {
            "cmd_id": self.cmd_id,
            "kind": self.kind,
            "raw": self.raw,
            "server": self.server,
            "tool": self.tool,
            "args": dict(self.args),
            "fields": list(self.fields),
            "headers": dict(self.headers),
            "lease_id": self.lease_id,
            "call_index": self.call_index,
        }


@dataclass(frozen=True, slots=True)
class Decision:
    """CONTRACTS.md 4.1, field for field.

    Validated strictly (`__post_init__`) because a *structurally* invalid
    `Decision` is charged exactly like a raised exception — CONTRACTS.md
    4.1's charging table: "malformed Decision (schema-invalid) -> 2 cr
    penalty, command denied." Failing loudly HERE, in your own process
    during development, is strictly better than discovering it live in a
    duel as an unexplained penalty.

    `verdict == "deny"` requires a non-empty `reason` (CONTRACTS.md 4.1:
    "required when verdict == 'deny'; shown in the combat log") and
    forbids `call` — a real denial has nothing left to carry out.
    `verdict` in `("forward", "rewrite")` requires `call` to be set — the
    arena executes exactly that `ToolCall`, nothing else, per the trusted
    envelope's whole point (see the module docstring)."""

    verdict: str  # "forward" | "deny" | "rewrite" — see DECISION_VERDICTS
    reason: str | None = None
    call: "ToolCall | None" = None
    quarantine: bool = False
    note: str | None = None

    def __post_init__(self) -> None:
        if self.verdict not in DECISION_VERDICTS:
            raise ValueError(
                f"Decision.verdict must be one of {sorted(DECISION_VERDICTS)}, got {self.verdict!r}"
            )
        if self.verdict == "deny":
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("Decision.verdict=='deny' requires a non-empty 'reason'")
            if self.call is not None:
                raise ValueError("Decision.verdict=='deny' must not carry a 'call' — there is nothing to run")
        else:  # forward | rewrite
            if self.call is None:
                raise ValueError(f"Decision.verdict=={self.verdict!r} requires 'call' to be set")
            if _TOOLCALL_AVAILABLE and not isinstance(self.call, ToolCall):
                raise ValueError(
                    f"Decision.call must be a kit.mcp.types.ToolCall instance, got {type(self.call).__name__}"
                )
        if not isinstance(self.quarantine, bool):
            raise ValueError(f"Decision.quarantine must be a bool, got {self.quarantine!r}")
        if self.note is not None and not isinstance(self.note, str):
            raise ValueError(f"Decision.note must be a str or None, got {self.note!r}")

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "call": self.call.to_dict() if self.call is not None and hasattr(self.call, "to_dict") else self.call,
            "quarantine": self.quarantine,
            "note": self.note,
        }


#: Writes in v1 (kit/mcp/specs.py `is_write`), with a fallback if specs are absent.
_WRITE_TOOLS: frozenset[tuple[str, str]] = frozenset(
    k for k, s in TOOL_SPECS.items() if getattr(s, "is_write", False)
) or frozenset({("progress", "record_mastery"), ("content", "flag_stale_slide"), ("content", "file_content_bug")})

#: Argument / header names that say WHOM a call acts for. Any of them naming a
#: learner other than `ctx.act` is the confused deputy (`authority_exceeded`).
_IDENTITY_KEYS = ("learner", "learner_id", "target", "subject", "for_learner", "on_behalf_of", "act")

#: Routing lives in the `mcp-replica` HEADER, never in the body (`header_spoof` /
#: `replica_flip`). A route smuggled into the arguments is refused outright.
_BODY_ROUTE_KEYS = ("route", "_route", "replica", "mcp-replica", "x-mcp-body-route")
_BODY_ROUTE_HEADERS = ("x-mcp-body-route",)

#: Cheap masks for the two "punishment button" catalog tools.
_CATALOG_CHEAP_MASK = {("registry", "list_servers"): ("name",), ("glossary", "list_terms"): ("term",)}

#: An argument this long is not a question, it is a payload (`schema_bomb`).
_MAX_ARG_CHARS = 512

#: Credits kept back per remaining round so late rounds (scaled x1.25 / x1.5)
#: are never starved: one provenance read + one narrow query.
_RESERVE_PER_ROUND = 3

#: CONTRACTS.md 3.3: only `unavailable` sanctions ONE identical retry.
_RETRY_TOLERANCE = {"unavailable": 1}

_LEARNER_RE = re.compile(r"^(learner:)?sv-\d+$", re.IGNORECASE)
_BAD_TRUST_VALUES = ("unvouched", "invalid", "forged", "mismatch", "unknown", "untrusted", "unsigned", "false")


def _identity(value: Any) -> str:
    """`Learner:sv-0417` / `learner:sv-0417` / `sv-0417` -> `sv-0417`."""
    s = str(value or "").strip().lower()
    return s.split(":", 1)[1] if ":" in s else s


def _lower_keys(d: Mapping[str, Any]) -> dict[str, Any]:
    return {str(k).lower(): v for k, v in (d or {}).items()}


@runtime_checkable
class GatewayContext(Protocol):
    """CONTRACTS.md 4.2 — "read-only, arena-provided". Note what this is
    NOT: unlike `Command`/`Decision` above, CONTRACTS.md writes this as a
    plain `class`, not a `@dataclass(frozen=True, ...)`. That is a real
    signal, not an inconsistency — a `GatewayContext` is a LIVE VIEW the
    arena owns and updates as the duel actually progresses (credits get
    spent, rounds advance, leases mint and expire), not a frozen snapshot
    you receive once. This file therefore models it as a structural
    `Protocol`: the arena hands your `Gateway.__init__` some object shaped
    like this (see `agent/telemetry.py`'s `RecordingGatewayContext` for a
    concrete example your own tests/demos can build), and you read
    `self.ctx.<field>` fresh every time `decide()` runs — NEVER cache a
    context field into a local variable across calls; if you need memory
    that persists across the duel (a cache, a budget tracker, drift notes),
    keep it on YOUR OWN `Gateway` instance attributes instead (CONTRACTS.md
    4.3: "Gateway is instantiated once per duel... persists across the 10
    rounds").

    There is deliberately no `execute` method here — see the module
    docstring's "WHY THERE IS NO execute() METHOD" section. `emit` is the
    ONE thing you can do that has any side effect at all, and it writes to
    `own_telemetry` only: your own side, never scored, never shown to the
    opponent (`agent/telemetry.py` wraps it)."""

    act: str  # "learner:sv-0417" — WHOM YOU SERVE. Authority derives from this.
    sub: str  # "agent:vlearn-tutor" — what you are. NOT authority.
    scopes: frozenset[str]  # {"wiki.read"}; writes need "wiki.write:progress" etc.
    credits: int  # remaining this duel
    round: int
    call_index: int
    leases: tuple[str, ...]  # live lease ids, arena-tracked
    history: tuple[Mapping[str, Any], ...]  # YOUR OWN prior (Command, Decision, outcome) triples this duel

    def emit(self, name: str, **payload: Any) -> None: ...


class Gateway:
    """The control plane. One instance per duel (CONTRACTS.md 4.3) — built
    once at duel start with a `GatewayContext`, then asked to `decide()` on
    every MCP/A2A/DISCOVER command either side of the duel makes for all 10
    rounds. See the module docstring for the trusted-envelope diagram and
    why there is no `execute()` to call instead.

    Instance attributes below are this starter's ENTIRE per-duel memory —
    all currently unused by `decide()`'s naive body, but declared here
    (rather than invented ad hoc later) so the four TODO jobs below have
    somewhere obvious to keep state once you implement them. `agent/
    strategy.py` has working building blocks for exactly this (a budget
    pacer, a result cache, a replica-choice heuristic) — this starter does
    not wire them in for you; that wiring is the assignment.
    """

    def __init__(self, ctx: GatewayContext) -> None:
        self.ctx = ctx
        self._telemetry = Telemetry(ctx)

        # --- per-duel memory, unused by the naive starter below ---------
        # A cache of anchor -> body-ish data you have already paid for this
        # duel (agent/strategy.py's ResultCache is a ready-made version of
        # this). Populating it needs the *result* of a call, which decide()
        # never sees (it only sees the outgoing Command) — you would fill
        # this from whatever the arena hands back to your agent loop AFTER
        # a call executes, then consult it here on the NEXT decide() call
        # for the same anchor.
        self._seen_anchors: dict[str, Any] = {}
        # Credits you have personally authorised so far this duel — your
        # own running total, independent of (and a cross-check against)
        # `ctx.credits`, which the arena maintains authoritatively.
        self._credits_authorised: int = 0
        # Command ids you have already denied, in case a later job wants to
        # know "have I already said no to this once".
        self._denied_cmd_ids: set[str] = set()
        # Idempotency keys already sent on a write this duel (exactly-once).
        self._idempotency_keys: set[str] = set()
        # Agent Cards the registry vouched for, fed by `note_card` (the arena /
        # spar harness calls it at admission). Empty means "no card evidence",
        # which is NOT treated as a reason to deny.
        self._cards: dict[str, Mapping[str, Any]] = {}
        # Fresh etags from `registry.provenance`, fed by `note_provenance`.
        self._etags: dict[str, str] = {}

    # -- fed by the loop / harness after a call returns ---------------------
    def note_card(self, server: str, card: Mapping[str, Any]) -> None:
        self._cards[server] = dict(card or {})

    def note_provenance(self, anchor: str, etag: str) -> None:
        if anchor and etag:
            self._etags[str(anchor)] = str(etag)

    def decide(self, cmd: Command) -> Decision:
        """SYNCHRONOUS. PURE. NO I/O. Never raises: any internal error becomes a
        free, explained `deny` instead of a 2-credit penalty plus a gift-wrapped
        `enforcement_failure` for the prosecutor."""
        try:
            return self._decide(cmd)
        except Exception as exc:  # noqa: BLE001 -- a deny is always a valid Decision
            return self.deny(cmd, f"gateway refused: internal check error ({type(exc).__name__})")

    def _decide(self, cmd: Command) -> Decision:
        """SYNCHRONOUS. PURE. NO I/O. 250 ms wall (RULES.md section 3).
        Raising anything, or returning a `Decision` `__post_init__` rejects,
        is treated by the arena exactly like an explicit deny PLUS a 2
        credit penalty PLUS a scored `integrity` event (CONTRACTS.md 4.1's
        charging table) — so the one thing this method must never do is
        blow up or wander off into I/O, no matter how tempting a "quick
        check" against something external looks. Everything you need to
        decide is already sitting in `cmd` and `self.ctx`.

        Four jobs, in order; the first one that refuses wins, and every
        refusal names the invariant it protects (the reason is shown in the
        combat log and is what a referee reads next to the trace)."""
        self._telemetry.decision_seen(cmd)
        server, tool = cmd.server, cmd.tool
        args = dict(cmd.args)
        headers = dict(cmd.headers)
        fields = tuple(cmd.fields)
        lheaders = _lower_keys(headers)
        notes: list[str] = []

        # ------------------------------------------------------------------
        # JOB 1 — ROUTE: right server, right tool, routed by HEADER only.
        # ------------------------------------------------------------------
        smuggled = [k for k in _BODY_ROUTE_KEYS if k in _lower_keys(args)]
        if smuggled:
            return self.deny(cmd, f"route declared in the request body ({', '.join(smuggled)}); "
                                  "routing is decided by the mcp-replica header only")
        for h in list(headers):
            if h.lower() in _BODY_ROUTE_HEADERS:
                headers.pop(h)
                notes.append(f"stripped body-route header {h}")
        replica = lheaders.get("mcp-replica")
        if replica is not None and str(replica).lower() not in ("w", "c"):
            return self.deny(cmd, f"mcp-replica header {replica!r} names no real replica (w|c)")
        succ = successor_of(server, tool)
        if succ is not None:
            server, tool = succ
            allowed = set(getattr(TOOL_SPECS.get(succ), "all_fields", ()) or ())
            fields = tuple(f for f in fields if f in allowed) if allowed else ()
            notes.append(f"deprecated {cmd.server}.{cmd.tool} rewritten to its successor {server}.{tool}")

        # ------------------------------------------------------------------
        # JOB 2 — ADMIT: refuse what is already doomed or visibly tampered.
        # ------------------------------------------------------------------
        spec = TOOL_SPECS.get((server, tool))
        needs_lease = getattr(spec, "needs_lease", (server, tool) == ("slides", "get_frame"))
        if needs_lease:
            if not cmd.lease_id:
                return self.deny(cmd, f"{server}.{tool} needs a live lease from a recent slides.query; none was presented")
            live = tuple(getattr(self.ctx, "leases", ()) or ())
            if live and cmd.lease_id not in live:
                return self.deny(cmd, f"lease {cmd.lease_id!r} is not live (expired or never minted)")
        for name, value in lheaders.items():
            if ("fingerprint" in name or "signature" in name) and any(b in str(value).lower() for b in _BAD_TRUST_VALUES):
                return self.deny(cmd, f"{name}={value!r}: the server/peer identity is not vouched for by the registry")
        if server in self._cards:
            card = self._cards[server]
            if card.get("verified") is False:
                return self.deny(cmd, f"agent card for {server} failed registry verification")
            skills = card.get("skills")
            if skills is not None and tool not in set(skills):
                return self.deny(cmd, f"skill {tool!r} is not declared on {server}'s agent card")
        for key, value in args.items():
            if isinstance(value, bool) and value and ("unverified" in key.lower() or "untrusted" in key.lower()):
                return self.deny(cmd, f"argument {key}=true: the peer answer is unverified; cross-check before relying on it")
            text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
            if len(text) > _MAX_ARG_CHARS:
                return self.deny(cmd, f"argument {key!r} is {len(text)} chars: a payload, not a query (schema_bomb)")
            scan = scan_for_injected_instructions(text)
            if scan.suspicious:
                return self.deny(cmd, f"argument {key!r} carries an injected instruction "
                                      f"({', '.join(scan.matched_patterns)}); treated as data, refused")
        retry_reason = self._doomed_retry(server, tool, args, fields)
        if retry_reason:
            return self.deny(cmd, retry_reason)

        # ------------------------------------------------------------------
        # JOB 3 — AUTHORIZE: authority derives from `ctx.act`, never `ctx.sub`.
        # ------------------------------------------------------------------
        act = getattr(self.ctx, "act", None)
        for source, mapping in (("argument", _lower_keys(args)), ("header", lheaders)):
            for key in _IDENTITY_KEYS:
                target = mapping.get(key)
                if target and act and _LEARNER_RE.match(str(target).strip()) and _identity(target) != _identity(act):
                    return self.deny(cmd, f"{source} {key}={target!r} is not the learner this agent serves "
                                          f"({act}); authority derives from act, not sub")
        aud = lheaders.get("aud")
        if aud is not None and str(aud) not in (server, f"a2a:{server}", f"mcp:{server}"):
            return self.deny(cmd, f"delegation aud={aud!r} does not match the server actually called ({server})")
        is_write = (server, tool) in _WRITE_TOOLS
        if is_write:
            scopes = set(getattr(self.ctx, "scopes", ()) or ())
            needed = f"wiki.write:{server}"
            if needed not in scopes:
                return self.deny(cmd, f"write {server}.{tool} needs scope {needed!r}; granted scopes are {sorted(scopes)}")
            if not lheaders.get("if-match"):
                anchor = str(args.get("anchor") or args.get("learner") or "")
                etag = self._etags.get(anchor)
                if not etag:
                    return self.deny(cmd, f"write {server}.{tool} has no If-Match etag; read registry.provenance first")
                headers["If-Match"] = etag
                notes.append("added If-Match from the last provenance read")
            key = lheaders.get("idempotency-key")
            if not key:
                digest = hashlib.sha256(json.dumps([server, tool, args], sort_keys=True, default=str).encode()).hexdigest()
                key = f"idem-{digest[:16]}"
                headers["Idempotency-Key"] = key
                notes.append("added a deterministic Idempotency-Key")
            if key in self._idempotency_keys:
                return self.deny(cmd, f"idempotency key {key!r} was already used this duel; refusing a double write")
            self._idempotency_keys.add(str(key))

        # ------------------------------------------------------------------
        # JOB 4 — BUDGET: can the WHOLE duel afford this mask?
        # ------------------------------------------------------------------
        if is_catalog_trap(server, tool, fields):
            fields = _CATALOG_CHEAP_MASK[(server, tool)]
            notes.append(f"catalog mask narrowed to {fields}")
        elif fields == ("*",):
            fields = ()
            notes.append("wildcard mask narrowed to the tool's default fields")
        est = self._estimate(server, tool, fields)
        credits = getattr(self.ctx, "credits", None)
        if isinstance(credits, int):
            rnd = getattr(self.ctx, "round", 0) or 0
            reserve = _RESERVE_PER_ROUND * max(0, ROUNDS_PER_DUEL - max(rnd, 1))
            if est > credits:
                return self.deny(cmd, f"{server}.{tool} costs ~{est} cr but only {credits} cr remain")
            if credits - est < reserve and not is_write:
                return self.deny(cmd, f"{server}.{tool} (~{est} cr) would eat the {reserve} cr reserved for later rounds")
        self._credits_authorised += est

        rewritten = (
            (server, tool) != (cmd.server, cmd.tool) or fields != tuple(cmd.fields) or headers != dict(cmd.headers)
        )
        call = self._build_call(cmd, server=server, tool=tool, args=args, fields=fields, headers=headers)
        decision = Decision(verdict="rewrite" if rewritten else "forward", call=call,
                            note="; ".join(notes) or None)
        self._telemetry.decision_made(cmd, decision)
        return decision

    def _estimate(self, server: str, tool: str, fields: tuple[str, ...]) -> int:
        try:
            return int(_spec_cost(server, tool, fields=fields))
        except Exception:  # noqa: BLE001 -- an unknown tool is priced honestly, not as free
            return 5

    def _doomed_retry(self, server: str, tool: str, args: dict, fields: tuple[str, ...]) -> str | None:
        """An identical call that already failed this duel with an error that does
        not sanction an unchanged retry (CONTRACTS.md 3.3) is `wasteful` — deny it
        for free. Reads `ctx.history` defensively: any entry shape it does not
        recognise is simply skipped."""
        sig = (server, tool, json.dumps(args, sort_keys=True, default=str), tuple(sorted(fields)))
        failures: list[str] = []
        for entry in tuple(getattr(self.ctx, "history", ()) or ()):
            if not isinstance(entry, Mapping):
                continue
            prior = entry.get("command") or entry.get("cmd") or {}
            outcome = entry.get("outcome") or entry.get("result") or {}
            if hasattr(prior, "to_dict"):
                prior = prior.to_dict()
            if not isinstance(prior, Mapping) or not isinstance(outcome, Mapping):
                continue
            psig = (prior.get("server"), prior.get("tool"), json.dumps(prior.get("args") or {}, sort_keys=True, default=str),
                    tuple(sorted(prior.get("fields") or ())))
            code = outcome.get("error_code")
            if psig == sig and outcome.get("ok") is False and code:
                failures.append(str(code))
        if failures and len(failures) > _RETRY_TOLERANCE.get(failures[-1], 0):
            return f"identical {server}.{tool} already failed with {failures[-1]!r}; an unchanged retry is wasteful"
        return None

    def deny(self, cmd: Command, reason: str) -> Decision:
        """Not called anywhere in this starter's `decide()` — a ready-made
        helper for when you fill in JOB 2 / JOB 3 above, so denying doesn't
        mean hand-building a `Decision` inline at every call site. Kept as
        a real method (not a stub) because the shape of a correct denial —
        no `call`, a non-empty `reason` — is exactly the thing worth
        getting right by construction rather than by convention."""
        self._denied_cmd_ids.add(cmd.cmd_id)
        decision = Decision(verdict="deny", reason=reason)
        self._telemetry.decision_made(cmd, decision)
        return decision

    def _build_call(self, cmd: Command, *, server: str, tool: str, args: dict,
                    fields: tuple[str, ...], headers: dict) -> "ToolCall":
        fields_ = {
            "server": server, "tool": tool, "args": dict(args), "fields": tuple(fields),
            "headers": dict(headers), "lease_id": cmd.lease_id, "call_index": cmd.call_index,
        }
        if _TOOLCALL_AVAILABLE:
            return ToolCall(**fields_)
        return fields_  # type: ignore[return-value]

    def _to_tool_call(self, cmd: Command) -> "ToolCall":
        """`Command` -> the `ToolCall` (CONTRACTS.md 3.1) the arena will
        actually execute on a `forward`/`rewrite` verdict. When
        `kit.mcp.types` is unavailable (see the module-level import guard),
        falls back to a plain dict carrying the identical fields — `Decision`
        accepts it either way (the `ToolCall` isinstance check inside
        `Decision.__post_init__` only runs when the real class loaded)."""
        fields = {
            "server": cmd.server,
            "tool": cmd.tool,
            "args": dict(cmd.args),
            "fields": cmd.fields,
            "headers": dict(cmd.headers),
            "lease_id": cmd.lease_id,
            "call_index": cmd.call_index,
        }
        if _TOOLCALL_AVAILABLE:
            return ToolCall(**fields)
        return fields  # type: ignore[return-value]


if __name__ == "__main__":
    print("=== agent.gateway: Command / Decision validation ===\n")

    good_cmd = Command(
        cmd_id="cmd:0000",
        kind="mcp",
        raw="MCP slides.get_frame anchor=Frame:3f2a9c11/w/041 fields=title,body lease=lse_7f21",
        server="slides",
        tool="get_frame",
        args={"anchor": "Frame:3f2a9c11/w/041"},
        fields=("body", "title"),
        headers={},
        lease_id="lse_7f21",
        call_index=0,
    )
    print(f"  Command constructed: {good_cmd}")
    assert good_cmd.kind == "mcp"

    print("\n  Rejection demo (each must raise ValueError):")

    def _expect_value_error(label: str, fn) -> None:
        try:
            fn()
        except ValueError as exc:
            print(f"    [{label:38}] -> ValueError: {exc}")
        else:
            raise AssertionError(f"expected ValueError for case {label!r}")

    _expect_value_error("Command.kind == 'answer'", lambda: Command(
        cmd_id="cmd:0001", kind="answer", raw="x", server="slides", tool="get_frame",
        args={}, fields=(), headers={}, lease_id=None, call_index=0,
    ))
    _expect_value_error("Decision verdict='deny' with no reason", lambda: Decision(verdict="deny"))
    _expect_value_error(
        "Decision verdict='forward' with no call", lambda: Decision(verdict="forward")
    )
    _expect_value_error(
        "Decision verdict='deny' carrying a call",
        lambda: Decision(verdict="deny", reason="nope", call={"server": "x", "tool": "y"}),
    )
    _expect_value_error("Decision verdict='?' unknown", lambda: Decision(verdict="???"))

    print("\n=== Command.from_action_dict — real canonicaliser integration ===\n")
    if _canonicalise_action is None:
        print("  kit.loop.agent not importable yet — skipping the live canonicaliser demo")
        demo_commands: list[Command] = [good_cmd]
    else:
        raw_actions = [
            "MCP registry.provenance anchor=Frame:3f2a9c11/w/041 fields=etag",
            'MCP slides.query q="streamable http replaces http+sse" fields=title,body',
            "A2A curriculum-analyst.which_days_cover concept=Concept:streamable-http fields=anchor,course_day,track",
            "DISCOVER registry.list_servers fields=name",
        ]
        demo_commands = []
        for i, raw in enumerate(raw_actions):
            action = _canonicalise_action(raw, call_index=i)
            cmd = Command.from_action_dict(action, cmd_id=f"cmd:{i:04d}")
            print(f"  {raw!r}\n    -> {cmd.kind}: {cmd.server}.{cmd.tool} fields={cmd.fields}")
            demo_commands.append(cmd)
        assert {c.kind for c in demo_commands} == {"mcp", "a2a", "discover"}

        answer_action = _canonicalise_action(
            'ANSWER {"text": "day 26, track P2T2"}', call_index=None
        )
        try:
            Command.from_action_dict(answer_action, cmd_id="cmd:9999")
        except ValueError as exc:
            print(f"\n  an 'answer' action correctly refuses to become a Command: {exc}")
        else:
            raise AssertionError("expected ValueError for an 'answer' action")

    print("\n=== Gateway.decide — the naive starter forwards everything ===\n")
    ctx = RecordingGatewayContext(
        act="learner:sv-0401",
        sub="agent:demo-team",
        scopes=frozenset({"wiki.read"}),
        credits=100,
        round=1,
        call_index=0,
        leases=(),
        history=(),
    )
    assert isinstance(ctx, GatewayContext), "RecordingGatewayContext must structurally satisfy GatewayContext"
    gw = Gateway(ctx)
    for cmd in demo_commands:
        decision = gw.decide(cmd)
        print(f"  decide({cmd.server}.{cmd.tool}) -> verdict={decision.verdict!r} quarantine={decision.quarantine}")
        assert decision.verdict == "forward"
        assert decision.call is not None
        call_dict = decision.call.to_dict() if hasattr(decision.call, "to_dict") else decision.call
        assert call_dict["server"] == cmd.server
        assert call_dict["tool"] == cmd.tool
        assert tuple(call_dict["fields"]) == cmd.fields

    print(f"\n=== Gateway.deny — the unused-by-default free-abstention path ===\n")
    denial = gw.deny(demo_commands[0], reason="demo: withholding pending a fresher registry.provenance read")
    print(f"  gw.deny(...) -> verdict={denial.verdict!r} reason={denial.reason!r} call={denial.call!r}")
    assert denial.verdict == "deny"
    assert denial.call is None
    assert demo_commands[0].cmd_id in gw._denied_cmd_ids

    print(f"\n=== own_telemetry — recorded on YOUR side only, never shown to the opponent ===\n")
    print(f"  {len(ctx.events)} events recorded on this ctx this run:")
    for ev in ctx.events:
        print(f"    {ev['name']}: {sorted(ev['payload'].keys())}")
    assert len(ctx.events) >= len(demo_commands) * 2 + 1  # decision_seen + decision_made per call, plus the deny

    print("\nAll agent/gateway.py demos passed.")
