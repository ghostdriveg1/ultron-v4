"""
packages/memory/lifecycle.py

EverMemOS-inspired memory lifecycle engine for Ultron V4.
Pattern source: EverMind-AI/EverOS + BAI-LAB/MemoryOS (mid_term.py heat model).

Lifecycle:
  raw text → MemCell (atomic episode)
  MemCell  → MemScene (thematic cluster, heat-based)
  MemScene → Foresight ("Ghost will need X next" prediction)
  Foresight → rd_loop.py (autonomous R&D trigger)

Heat formula (MemoryOS):
  H = α*N_visit + β*L_interaction + γ*R_recency
  R_recency = exp(-(now - last_visit) / TAU_HOURS)

Pre-registered bugs:
  LC1 [HIGH]  Foresight LLM call fails → return cached or empty, never crash worker
  LC2 [HIGH]  Scene clustering with no embedding → skip cluster, add to scene-less pool
  LC3 [MED]   Heat rebuild on every ingest is O(N) → cap scene count per user at 200
  LC4 [MED]   Foresight TTL not set in Redis → stale predictions → set 6h TTL always
  LC5 [LOW]   Cell eviction during concurrent ingests → asyncio.Lock per user_id
"""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional

# Heat constants (tunable)
HEAT_ALPHA = 1.0   # N_visit weight
HEAT_BETA  = 0.5   # L_interaction (cell count) weight
HEAT_GAMMA = 2.0   # recency weight
RECENCY_TAU_HOURS = 24.0   # decay half-life

MAX_SCENES_PER_USER = 200  # LC3
FORESIGHT_TTL_SECONDS = 6 * 3600  # LC4 — 6h Redis TTL
CELL_WINDOW = 20   # max cells in STM per user (Redis list)


# ─────────────────────────────────────────────
# Time helpers
# ─────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _recency(last_visit_iso: str) -> float:
    """Exponential decay R = exp(-Δt / τ). Returns 0..1."""
    try:
        last = datetime.fromisoformat(last_visit_iso)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        delta_h = (datetime.now(timezone.utc) - last).total_seconds() / 3600.0
        return math.exp(-delta_h / RECENCY_TAU_HOURS)
    except Exception:
        return 1.0


def _heat(n_visit: int, l_interaction: int, last_visit_iso: str) -> float:
    R = _recency(last_visit_iso)
    return HEAT_ALPHA * n_visit + HEAT_BETA * l_interaction + HEAT_GAMMA * R


# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────

@dataclass
class MemCell:
    """Atomic episode — raw text, never paraphrased at write time."""
    cell_id:    str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id:    str = ""
    channel_id: str = ""
    raw_text:   str = ""       # verbatim message(s)
    timestamp:  str = field(default_factory=_now_iso)
    scene_id:   Optional[str] = None
    heat:       float = 1.0
    metadata:   Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "MemCell":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MemScene:
    """Thematic cluster of MemCells. Equivalent to MemoryOS session."""
    scene_id:       str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id:        str = ""
    topic:          str = ""   # LLM-extracted topic label
    summary:        str = ""   # LLM-generated on promotion
    cell_ids:       List[str] = field(default_factory=list)
    created_at:     str = field(default_factory=_now_iso)
    last_visited:   str = field(default_factory=_now_iso)
    n_visit:        int = 0
    heat:           float = 1.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "MemScene":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def recompute_heat(self) -> None:
        self.heat = _heat(self.n_visit, len(self.cell_ids), self.last_visited)


@dataclass
class Foresight:
    """EverMemOS Foresight: time-bounded predictions of what Ghost needs next."""
    foresight_id:  str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id:       str = ""
    predictions:   List[str] = field(default_factory=list)  # ranked list
    generated_at:  str = field(default_factory=_now_iso)
    valid_until:   str = ""
    context_cells: List[str] = field(default_factory=list)  # cell_ids used

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "Foresight":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def is_valid(self) -> bool:
        try:
            vt = datetime.fromisoformat(self.valid_until)
            if vt.tzinfo is None:
                vt = vt.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) < vt
        except Exception:
            return False


# ─────────────────────────────────────────────
# LifecycleEngine
# ─────────────────────────────────────────────

class LifecycleEngine:
    """
    Manages full memory lifecycle per user.

    Redis key schema:
      lifecycle:cell:{user_id}:{cell_id}  → MemCell JSON
      lifecycle:stm:{user_id}             → Redis list of cell_ids (20-item window)
      lifecycle:scene:{user_id}:{scene_id}→ MemScene JSON
      lifecycle:scene_index:{user_id}     → JSON list of scene_ids
      lifecycle:foresight:{user_id}       → Foresight JSON (TTL 6h)
    """

    def __init__(self, redis_client):
        self.redis = redis_client
        self._locks: Dict[str, asyncio.Lock] = {}   # per-user locks (LC5)

    def _lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    # ── STM (Redis list) ─────────────────────

    async def ingest(self, user_id: str, channel_id: str, raw_text: str,
                     metadata: Optional[Dict] = None) -> str:
        """Write raw text → MemCell → STM window. Returns cell_id."""
        cell = MemCell(
            user_id=user_id,
            channel_id=channel_id,
            raw_text=raw_text,
            metadata=metadata or {},
        )
        async with self._lock(user_id):
            # persist cell
            cell_key = f"lifecycle:cell:{user_id}:{cell.cell_id}"
            await self.redis.set(cell_key, json.dumps(cell.to_dict()))

            # push to STM list, trim to CELL_WINDOW
            stm_key = f"lifecycle:stm:{user_id}"
            pipe = self.redis.pipeline()
            pipe.rpush(stm_key, cell.cell_id)
            pipe.ltrim(stm_key, -CELL_WINDOW, -1)
            await pipe.execute()

        return cell.cell_id

    async def get_stm(self, user_id: str) -> List[MemCell]:
        """Return all MemCells in STM window."""
        stm_key = f"lifecycle:stm:{user_id}"
        cell_ids = await self.redis.lrange(stm_key, 0, -1)
        cells = []
        for cid in cell_ids:
            raw = await self.redis.get(f"lifecycle:cell:{user_id}:{cid}")
            if raw:
                cells.append(MemCell.from_dict(json.loads(raw)))
        return cells

    # ── MTM (MemScene clusters) ──────────────

    async def promote_to_scene(
        self,
        user_id: str,
        topic: str,
        summary: str,
        cell_ids: List[str],
    ) -> str:
        """Promote a group of cells into a named MemScene. Returns scene_id."""
        scene = MemScene(
            user_id=user_id,
            topic=topic,
            summary=summary,
            cell_ids=cell_ids,
        )
        async with self._lock(user_id):
            scene_key = f"lifecycle:scene:{user_id}:{scene.scene_id}"
            await self.redis.set(scene_key, json.dumps(scene.to_dict()))
            await self._append_scene_index(user_id, scene.scene_id)
        return scene.scene_id

    async def get_scene(self, user_id: str, scene_id: str) -> Optional[MemScene]:
        raw = await self.redis.get(f"lifecycle:scene:{user_id}:{scene_id}")
        return MemScene.from_dict(json.loads(raw)) if raw else None

    async def visit_scene(self, user_id: str, scene_id: str) -> None:
        """Increment N_visit, update heat. Called on retrieval hit."""
        scene = await self.get_scene(user_id, scene_id)
        if not scene:
            return
        scene.n_visit += 1
        scene.last_visited = _now_iso()
        scene.recompute_heat()
        await self.redis.set(
            f"lifecycle:scene:{user_id}:{scene_id}",
            json.dumps(scene.to_dict()),
        )

    async def list_scenes(self, user_id: str) -> List[MemScene]:
        """Return all scenes for user, sorted by heat descending."""
        index = await self._get_scene_index(user_id)
        scenes = []
        for sid in index:
            s = await self.get_scene(user_id, sid)
            if s:
                s.recompute_heat()
                scenes.append(s)
        scenes.sort(key=lambda x: x.heat, reverse=True)
        return scenes

    async def evict_cold_scenes(self, user_id: str) -> int:
        """Remove scenes beyond MAX_SCENES_PER_USER by lowest heat (LFU analog). LC3."""
        scenes = await self.list_scenes(user_id)  # already sorted hot→cold
        if len(scenes) <= MAX_SCENES_PER_USER:
            return 0
        to_evict = scenes[MAX_SCENES_PER_USER:]
        for s in to_evict:
            await self.redis.delete(f"lifecycle:scene:{user_id}:{s.scene_id}")
        kept = [s.scene_id for s in scenes[:MAX_SCENES_PER_USER]]
        await self.redis.set(
            f"lifecycle:scene_index:{user_id}", json.dumps(kept)
        )
        return len(to_evict)

    # ── Foresight ────────────────────────────

    async def generate_foresight(
        self,
        user_id: str,
        llm_fn: Callable,   # async fn(messages) → str
    ) -> Foresight:
        """
        Generate Foresight: what Ghost will need next.
        Uses hot scenes + recent STM cells as context.
        LC1: LLM failure returns cached or empty Foresight, never raises.
        """
        # check cache first
        existing = await self.get_foresight(user_id)
        if existing and existing.is_valid():
            return existing

        # build context
        scenes = (await self.list_scenes(user_id))[:5]  # top 5 hottest
        cells = await self.get_stm(user_id)
        recent_texts = [c.raw_text for c in cells[-5:]]
        scene_summaries = [f"[{s.topic}] {s.summary}" for s in scenes]

        context = "\n".join(scene_summaries + recent_texts)
        prompt = [
            {"role": "system", "content": (
                "You are Ultron's memory foresight engine. "
                "Based on recent activity, predict what the user will likely need or ask about next. "
                "Return a JSON list of 3-5 short predictions, most likely first. "
                "Example: [\"ChemE calculator improvements\", \"PDF export feature\", \"Deploy voice space\"]"
            )},
            {"role": "user", "content": f"Recent activity:\n{context}\n\nPredict next 3-5 needs:"}
        ]

        predictions: List[str] = []
        try:
            raw_response = await llm_fn(prompt)
            # strip markdown fences
            clean = raw_response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = json.loads(clean)
            if isinstance(parsed, list):
                predictions = [str(p) for p in parsed[:5]]
        except Exception:
            pass  # LC1: never crash

        valid_until = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
        foresight = Foresight(
            user_id=user_id,
            predictions=predictions,
            valid_until=valid_until,
            context_cells=[c.cell_id for c in cells],
        )

        # persist with TTL (LC4)
        foresight_key = f"lifecycle:foresight:{user_id}"
        await self.redis.set(
            foresight_key,
            json.dumps(foresight.to_dict()),
            ex=FORESIGHT_TTL_SECONDS,
        )
        return foresight

    async def get_foresight(self, user_id: str) -> Optional[Foresight]:
        raw = await self.redis.get(f"lifecycle:foresight:{user_id}")
        if not raw:
            return None
        try:
            return Foresight.from_dict(json.loads(raw))
        except Exception:
            return None

    # ── Index helpers ────────────────────────

    async def _get_scene_index(self, user_id: str) -> List[str]:
        raw = await self.redis.get(f"lifecycle:scene_index:{user_id}")
        return json.loads(raw) if raw else []

    async def _append_scene_index(self, user_id: str, scene_id: str) -> None:
        index = await self._get_scene_index(user_id)
        if scene_id not in index:
            index.append(scene_id)
        await self.redis.set(
            f"lifecycle:scene_index:{user_id}", json.dumps(index)
        )
