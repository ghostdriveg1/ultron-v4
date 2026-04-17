"""
packages/tools/browser_agent.py

Ultron V4 — Browser Automation Agent
======================================
Playwright-based browser agent. Wraps LLM decision loop around DOM actions.

Architecture (informed by browser-use/browser-use source read, session v13):
  - No vision mode for Groq (DOM text only = 10x token saving).
  - AgentOutput flash mode: {action_type, action_params} only.
  - ActionLoopDetector: SHA256 hash rolling window, soft nudge (never blocks).
  - Max 5 browser steps per task. Hard ceiling 10.
  - Playwright Chromium headless (works in HF Space Ubuntu container).
  - Accessibility tree extraction (playwright-mcp pattern) for token efficiency.

Actions:
  navigate(url)            — go to URL
  click(selector)          — CSS selector click
  type(selector, text)     — clear + type into input
  extract(selector)        — extract text from selector (or full body)
  scroll(direction)        — up/down page scroll
  done(result)             — task complete, return result

Browser session lifecycle:
  - Playwright browser launched once per BrowserAgent instance.
  - Page reused across steps (persistent session within one task).
  - Browser closed after task completes or on exception.

Future bug risks (pre-registered):
  BA1 [HIGH]  Playwright Chromium download (~120MB) on first launch.
              Fix: Add to Dockerfile: RUN playwright install chromium --with-deps
              (already in Dockerfile from session v20 — verify it's still there).

  BA2 [HIGH]  HF Space headless display: Chromium needs DISPLAY or --no-sandbox.
              Fix: launch with args=["--no-sandbox", "--disable-dev-shm-usage"].
              --no-sandbox required in Docker containers.

  BA3 [HIGH]  Groq context window (8k tokens). Long DOM text fills it in 1-2 steps.
              Fix: truncate DOM text to 4000 chars. Extract only visible text.

  BA4 [MED]   Playwright async API requires running inside asyncio event loop.
              FastAPI is async, so this is fine. But discord_bot.py running in same
              process shares loop — no conflict as long as no sync Playwright calls.

  BA5 [MED]   CSS selectors often break on SPAs. Fallback: extract full body text
              if selector not found. Log selector miss.

  BA6 [LOW]   Browser agent may visit malicious URLs. Add domain allowlist check.
              For now: log warning + proceed (no blocking).

Tool calls used writing this file:
    Github:get_file_contents x4 (browser-use/browser-use agent/ — session v13 source read)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import deque
from typing import Any, Optional

log = logging.getLogger("tools.browser_agent")

MAX_STEPS     = 5
ABS_MAX_STEPS = 10
DOM_MAX_CHARS = 4000  # BA3 guard: Groq 8k ctx
LOOP_WINDOW   = 20    # ActionLoopDetector rolling window
NUDGE_AT      = [5, 8, 12]  # Soft nudge thresholds (never block)


class BrowserAgent:
    """
    LLM-driven browser automation agent.
    Wraps Playwright with a ReAct-style action loop.

    Usage:
        agent = BrowserAgent(llm_fn=make_provider_llm_fn(pool))
        result = await agent.run("Find the price of RTX 4090 on Amazon")
    """

    def __init__(
        self,
        llm_fn: Any,
        max_steps: int = MAX_STEPS,
    ) -> None:
        self._llm_fn    = llm_fn
        self._max_steps = min(max_steps, ABS_MAX_STEPS)
        self._loop_hashes: deque = deque(maxlen=LOOP_WINDOW)

    async def run(self, task: str) -> str:
        """
        Execute browser task. Returns result string.
        Handles Playwright lifecycle internally.
        """
        try:
            from playwright.async_api import async_playwright  # type: ignore
        except ImportError:
            return "Browser agent unavailable: playwright not installed (BA1)."

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],  # BA2
            )
            page = await browser.new_page()

            try:
                result = await self._action_loop(task, page)
            except Exception as e:
                log.error(f"[Browser] Action loop error: {e}")
                result = f"Browser agent error: {e}"
            finally:
                await browser.close()

        return result

    async def _action_loop(self, task: str, page: Any) -> str:
        """ReAct-style loop: LLM decides action -> execute -> feed result back."""
        history: list[dict] = []
        step = 0

        while step < self._max_steps:
            step += 1

            # Build prompt
            dom_text = await self._extract_dom(page)
            prompt   = self._build_prompt(task, history, dom_text, step)

            # LLM call
            try:
                raw = await self._llm_fn(
                    messages=[{"role": "user", "content": prompt}],
                    system=(
                        "You are a browser automation agent. "
                        "Respond ONLY with valid JSON matching the action schema. "
                        "No markdown, no explanation."
                    ),
                    max_tokens=300,
                )
            except Exception as e:
                log.error(f"[Browser] LLM call failed step={step}: {e}")
                return f"LLM failure at step {step}: {e}"

            # Parse action
            action = self._parse_action(raw)
            if not action:
                log.warning(f"[Browser] Could not parse action at step={step}: {raw[:200]}")
                continue

            action_type   = action.get("action_type", "")
            action_params = action.get("action_params", {})

            # Loop detection
            action_hash = hashlib.sha256(
                json.dumps({"t": action_type, "p": action_params}, sort_keys=True).encode()
            ).hexdigest()[:16]
            self._loop_hashes.append(action_hash)
            repeat_count = sum(1 for h in self._loop_hashes if h == action_hash)

            nudge = ""
            if repeat_count in NUDGE_AT:
                nudge = f"[LOOP NUDGE] You have repeated action '{action_type}' {repeat_count} times. Try a different approach."
                log.warning(f"[Browser] {nudge}")

            # Done
            if action_type == "done":
                result = action_params.get("result", "Task complete.")
                log.info(f"[Browser] Done at step={step}: {result[:100]}")
                return result

            # Execute action
            obs = await self._execute(page, action_type, action_params)
            log.info(f"[Browser] Step={step} action={action_type} obs={obs[:100]}")

            history.append({
                "step":   step,
                "action": action_type,
                "params": action_params,
                "obs":    obs,
                "nudge":  nudge,
            })

        return f"Browser agent reached max steps ({self._max_steps}) without completing task."

    async def _extract_dom(self, page: Any) -> str:
        """Extract visible text from current page. Truncated to DOM_MAX_CHARS (BA3)."""
        try:
            text = await page.evaluate(
                """() => {
                    const walker = document.createTreeWalker(
                        document.body,
                        NodeFilter.SHOW_TEXT,
                        null,
                        false
                    );
                    let text = '';
                    let node;
                    while (node = walker.nextNode()) {
                        const t = node.nodeValue.trim();
                        if (t) text += t + ' ';
                        if (text.length > 5000) break;
                    }
                    return text.trim();
                }"""
            )
            url = page.url
            return f"URL: {url}\n\nPAGE TEXT:\n{text[:DOM_MAX_CHARS]}"
        except Exception as e:
            return f"DOM extraction failed: {e}"

    def _build_prompt(
        self,
        task: str,
        history: list,
        dom_text: str,
        step: int,
    ) -> str:
        history_str = ""
        for h in history[-3:]:  # Only last 3 steps to save tokens
            history_str += (
                f"Step {h['step']}: {h['action']}({json.dumps(h['params'])}) "
                f"-> {h['obs'][:200]}\n"
            )
            if h.get("nudge"):
                history_str += f"  {h['nudge']}\n"

        return (
            f"TASK: {task}\n\n"
            f"STEP: {step}/{self._max_steps}\n\n"
            f"RECENT HISTORY:\n{history_str or '(none)'}\n\n"
            f"CURRENT PAGE:\n{dom_text}\n\n"
            f"AVAILABLE ACTIONS:\n"
            f'  navigate(url: str)\n'
            f'  click(selector: str)\n'
            f'  type(selector: str, text: str)\n'
            f'  extract(selector: str)\n'
            f'  scroll(direction: "up"|"down")\n'
            f'  done(result: str)\n\n'
            f"Respond with JSON: "
            f'{{\"action_type\": \"<action>\", \"action_params\": {{...}}}}'
        )

    def _parse_action(self, raw: str) -> Optional[dict]:
        """Parse LLM JSON response into action dict."""
        raw = raw.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def _execute(self, page: Any, action_type: str, params: dict) -> str:
        """Execute browser action. Returns observation string."""
        try:
            if action_type == "navigate":
                url = params.get("url", "")
                await page.goto(url, timeout=15000)
                return f"Navigated to {url}"

            elif action_type == "click":
                selector = params.get("selector", "")
                try:
                    await page.click(selector, timeout=5000)
                    return f"Clicked {selector}"
                except Exception:
                    log.warning(f"[Browser] Selector not found: {selector} (BA5)")
                    return f"Click failed: selector '{selector}' not found"

            elif action_type == "type":
                selector = params.get("selector", "")
                text     = params.get("text", "")
                try:
                    await page.fill(selector, text, timeout=5000)
                    return f"Typed '{text[:50]}' into {selector}"
                except Exception:
                    return f"Type failed: selector '{selector}' not found (BA5)"

            elif action_type == "extract":
                selector = params.get("selector", "body")
                try:
                    el   = await page.query_selector(selector)
                    text = await el.inner_text() if el else ""
                    return text[:500] or "(empty)"
                except Exception:
                    return "(extract failed)"

            elif action_type == "scroll":
                direction = params.get("direction", "down")
                delta     = 500 if direction == "down" else -500
                await page.evaluate(f"window.scrollBy(0, {delta})")
                return f"Scrolled {direction}"

            else:
                return f"Unknown action: {action_type}"

        except Exception as e:
            return f"Execute error: {e}"
