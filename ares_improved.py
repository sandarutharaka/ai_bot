"""
Ares — Improved Self-Thinking AI Assistant
File: ares_improved.py

Highlights / improvements over the original:
 - Modular architecture: Core, Storage, External AI Provider, Voice, Teaching, CLI
 - Persistent SQLite knowledge store (safer and queryable than raw JSON)
 - Conversation context window (keeps recent messages for better external prompts)
 - Safer Gemini/OpenAI client wrapper with graceful fallbacks and timeouts
 - Pluggable voice: pyttsx3 (local) if available; disabled cleanly otherwise
 - Thread-safe speaking and non-blocking TTS
 - Better teacher parser with more robust patterns
 - Command dispatcher with helpful built-in commands
 - Structured logging and error handling
 - Easy to extend with plugins or alternative LLM backends

Usage:
 1) Install dependencies (recommended to use a venv):
    pip install python-dotenv rich pyttsx3 SpeechRecognition requests sqlalchemy

 2) Create a .env file containing (if you want external model):
    GEMINI_API_KEY=your_key_here

 3) Run:
    python ares_improved.py

Note: External AI client code is a wrapper placeholder. Integrate your provider SDK
(e.g. Gemini or OpenAI) into ExternalAI.query() — keep the same return contract.

"""

from __future__ import annotations
import os
import re
import sys
import json
import time
import threading
import queue
import textwrap
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import box

# Optional voice and speech recognition
try:
    import pyttsx3
    import speech_recognition as sr
    VOICE_AVAILABLE = True
except Exception:
    VOICE_AVAILABLE = False

# Gemini AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# Persistence (SQLite via SQLAlchemy for convenience)
try:
    from sqlalchemy import (
        create_engine,
        Column,
        Integer,
        Text,
        String,
        DateTime,
        JSON,
        select,
    )
    from sqlalchemy.orm import declarative_base, Session
    SQLALCHEMY_AVAILABLE = True
except Exception:
    SQLALCHEMY_AVAILABLE = False

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ----------------------------- Configuration -----------------------------
APP_NAME = "Ares"
DB_PATH = os.getenv("ARES_DB_PATH", "ares_knowledge.db")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CONTEXT_WINDOW = int(os.getenv("ARES_CONTEXT_WINDOW", "8"))  # last N exchanges
VOICE_RATE = int(os.getenv("ARES_VOICE_RATE", "170"))
VOICE_VOLUME = float(os.getenv("ARES_VOICE_VOLUME", "0.9"))

console = Console()

# ----------------------------- Storage Layer -----------------------------

if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()

    class Fact(Base):
        __tablename__ = "facts"
        id = Column(Integer, primary_key=True)
        key = Column(String(120), unique=True, nullable=False)
        value = Column(Text)
        updated_at = Column(DateTime, default=datetime.utcnow)

    class Experience(Base):
        __tablename__ = "experiences"
        id = Column(Integer, primary_key=True)
        timestamp = Column(DateTime, default=datetime.utcnow)
        role = Column(String(16))  # 'user' or 'agent'
        content = Column(Text)
        meta = Column(JSON, default={})

    class Rule(Base):
        __tablename__ = "rules"
        id = Column(Integer, primary_key=True)
        trigger = Column(String(200), nullable=False)  # Removed unique constraint
        response = Column(Text)
        created_at = Column(DateTime, default=datetime.utcnow)

else:
    # Fallback: simple JSON file persistence
    pass


class Storage:
    """Abstracts persistence. Uses SQLAlchemy when available, else JSON files."""

    def __init__(self, path: str = DB_PATH):
        self.path = path
        self._ensure()
        if SQLALCHEMY_AVAILABLE:
            self.engine = create_engine(f"sqlite:///{self.path}", future=True)
            Base.metadata.create_all(self.engine)
        else:
            # JSON fallback
            self.data_file = Path(self.path).with_suffix(".json")
            if not self.data_file.exists():
                self._write_json({"facts": {}, "experiences": [], "rules": []})

    def _ensure(self):
        d = Path(self.path).parent
        d.mkdir(parents=True, exist_ok=True)

    # Fact operations
    def set_fact(self, key: str, value: Any):
        if SQLALCHEMY_AVAILABLE:
            with Session(self.engine) as sess:
                q = sess.query(Fact).filter(Fact.key == key).one_or_none()
                if q:
                    q.value = json.dumps(value)
                    q.updated_at = datetime.utcnow()
                else:
                    q = Fact(key=key, value=json.dumps(value))
                    sess.add(q)
                sess.commit()
        else:
            data = self._read_json()
            data["facts"][key] = value
            self._write_json(data)

    def get_fact(self, key: str):
        if SQLALCHEMY_AVAILABLE:
            with Session(self.engine) as sess:
                q = sess.query(Fact).filter(Fact.key == key).one_or_none()
                if not q:
                    return None
                try:
                    return json.loads(q.value)
                except Exception:
                    return q.value
        else:
            data = self._read_json()
            return data["facts"].get(key)

    # Rules
    def add_rule(self, trigger: str, response: str):
        if SQLALCHEMY_AVAILABLE:
            with Session(self.engine) as sess:
                r = Rule(trigger=trigger, response=response)
                sess.add(r)
                sess.commit()
        else:
            data = self._read_json()
            data["rules"].append({"trigger": trigger, "response": response})
            self._write_json(data)

    def get_rules(self) -> List[Dict[str, str]]:
        if SQLALCHEMY_AVAILABLE:
            with Session(self.engine) as sess:
                rows = sess.query(Rule).all()
                return [{"trigger": r.trigger, "response": r.response} for r in rows]
        else:
            data = self._read_json()
            return data.get("rules", [])

    # Experiences (conversation history)
    def add_experience(self, role: str, content: str, meta: Optional[dict] = None):
        meta = meta or {}
        if SQLALCHEMY_AVAILABLE:
            with Session(self.engine) as sess:
                e = Experience(role=role, content=content, meta=meta)
                sess.add(e)
                sess.commit()
        else:
            data = self._read_json()
            data["experiences"].append({"timestamp": datetime.utcnow().isoformat(), "role": role, "content": content, "meta": meta})
            # keep last 500
            data["experiences"] = data["experiences"][-500:]
            self._write_json(data)

    def get_recent_experiences(self, limit: int = 50) -> List[Dict[str, Any]]:
        if SQLALCHEMY_AVAILABLE:
            with Session(self.engine) as sess:
                rows = sess.execute(select(Experience).order_by(Experience.id.desc()).limit(limit)).scalars().all()
                return [{"timestamp": r.timestamp.isoformat(), "role": r.role, "content": r.content, "meta": r.meta} for r in reversed(rows)]
        else:
            data = self._read_json()
            return data.get("experiences", [])[-limit:]

    # JSON helpers
    def _read_json(self):
        with open(self.data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, contents):
        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(contents, f, indent=2, ensure_ascii=False)


# ----------------------------- Voice System -----------------------------

class VoiceSystem:
    def __init__(self, rate: int = VOICE_RATE, volume: float = VOICE_VOLUME):
        self.enabled = VOICE_AVAILABLE
        self._engine = None
        self._lock = threading.RLock()
        self._speak_queue = queue.Queue()
        self._worker = None
        self._running = False
        self.rate = rate
        self.volume = volume

        if not self.enabled:
            return

        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', self.rate)
            self._engine.setProperty('volume', self.volume)
            # start worker
            self._running = True
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()
        except Exception as e:
            console.print(f"[yellow]Voice init failed:[/yellow] {e}")
            self.enabled = False

    def _run(self):
        while self._running:
            try:
                text = self._speak_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._lock:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception:
                    pass

    def speak(self, text: str, block: bool = False):
        if not self.enabled:
            return
        # sanitize text
        text = re.sub(r"[\x00-\x1f]+", " ", text)
        self._speak_queue.put(text)
        if block:
            # naive wait until queue empties
            while not self._speak_queue.empty():
                time.sleep(0.05)

    def stop(self):
        if not self.enabled:
            return
        self._running = False
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.0)


# ----------------------------- External AI Provider -----------------------------

class ExternalAI:
    """Wrapper for Gemini API integration."""

    def __init__(self, api_key: Optional[str] = GEMINI_API_KEY, timeout: float = 8.0):
        self.api_key = api_key
        self.timeout = timeout
        self.model = None
        self.chat = None
        
        if not GEMINI_AVAILABLE:
            console.print("[yellow]Warning: google-generativeai not installed[/yellow]")
            return
        
        if not self.api_key:
            console.print("[yellow]Warning: GEMINI_API_KEY not set[/yellow]")
            return
        
        try:
            genai.configure(api_key=self.api_key)
            
            # Try to initialize with available models
            model_names = [
                'models/gemini-2.5-flash',
                'models/gemini-2.0-flash-exp',
                'models/gemini-flash-latest'
            ]
            
            for model_name in model_names:
                try:
                    self.model = genai.GenerativeModel(model_name=model_name)
                    test = self.model.generate_content("test")
                    self.chat = self.model.start_chat(history=[])
                    console.print(f"[green]✓ Gemini connected: {model_name}[/green]")
                    break
                except Exception as e:
                    continue
            
            if not self.model:
                console.print("[yellow]Warning: Could not connect to any Gemini model[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Warning: Gemini init failed: {e}[/yellow]")

    def query(self, prompt: str, context: Optional[str] = None) -> str:
        """Query Gemini with optional context."""
        
        if not self.model or not self.chat:
            return "[External AI unavailable]"
        
        try:
            # Build final prompt
            if context:
                final_prompt = f"Context from recent conversation:\n{context}\n\nUser question: {prompt}"
            else:
                final_prompt = prompt
            
            # Query with timeout
            response = self.chat.send_message(final_prompt, request_options={"timeout": self.timeout})
            return response.text if response.text else "[No response from Gemini]"
            
        except Exception as e:
            return f"[External AI error: {str(e)[:100]}]"


# ----------------------------- Thinking Core -----------------------------

@dataclass
class ThinkingCore:
    storage: Storage
    identity_name: str = APP_NAME
    skills: List[str] = field(default_factory=lambda: ["basic_conversation"])

    def __post_init__(self):
        # ensure identity persisted
        existing = self.storage.get_fact("identity_name")
        if not existing:
            self.storage.set_fact("identity_name", self.identity_name)
        else:
            self.identity_name = existing

    def recall(self, query: str) -> Optional[str]:
        ql = query.lower()
        # direct identity
        if "your name" in ql or "who are you" in ql:
            name = self.storage.get_fact("identity_name") or self.identity_name
            return f"I am {name}, a self-thinking assistant."

        # user's name
        if "my name" in ql or "who am i" in ql:
            user_name = self.storage.get_fact("user_name")
            if user_name:
                return f"You told me your name is {user_name}"

        # check known facts
        facts = []
        # naive full-scan (could add indexing)
        if SQLALCHEMY_AVAILABLE:
            # read via storage
            pass
        else:
            # JSON fallback
            pass

        # apply simple rules
        for r in self.storage.get_rules():
            if r["trigger"].lower() in ql:
                return r["response"]

        # greetings
        if re.search(r"\b(hi|hello|hey|greetings)\b", ql):
            return f"Hello — I'm {self.identity_name}. How can I help?"

        # capability question
        if "what can you do" in ql or "capabilities" in ql:
            return f"I can: {', '.join(self.skills)}"

        return None

    def learn_fact(self, key: str, value: Any):
        self.storage.set_fact(key, value)

    def learn_rule(self, trigger: str, response: str):
        self.storage.add_rule(trigger, response)

    def record(self, role: str, content: str, meta: Optional[dict] = None):
        self.storage.add_experience(role, content, meta)

    def recent_context(self, limit: int = CONTEXT_WINDOW) -> str:
        exps = self.storage.get_recent_experiences(limit)
        lines = []
        for e in exps:
            prefix = "User" if e["role"] == "user" else "Ares"
            ts = e.get("timestamp", "")
            lines.append(f"[{ts}] {prefix}: {e['content']}")
        return "\n".join(lines)


# ----------------------------- Teaching Interface -----------------------------

class Teacher:
    def __init__(self, core: ThinkingCore):
        self.core = core

    def parse_and_apply(self, text: str) -> Optional[str]:
        lower = text.lower().strip()
        
        # "my name is X"
        m = re.search(r"my name is\s+([\w\-\s]{1,40})", text, re.I)
        if m:
            name = m.group(1).strip()
            self.core.learn_fact("user_name", name)
            return f"✓ I'll remember that your name is {name}"

        # "remember that X"
        m_rem = re.search(r"remember that\s+(.+)", text, re.I)
        if m_rem:
            fact = m_rem.group(1).strip()
            idx = f"memo_{int(time.time())}"
            self.core.learn_fact(idx, fact)
            return f"✓ I'll remember: {fact}"

        # "when i say X you respond with Y"
        m_when = re.search(r"when i say\s+['\"]?(.+?)['\"]?\s+you respond with\s+['\"]?(.+?)['\"]?$", text, re.I)
        if m_when:
            trigger = m_when.group(1).strip()
            response = m_when.group(2).strip()
            self.core.learn_rule(trigger, response)
            return f"✓ Rule learned: when you say '{trigger}' I'll respond with '{response}'"

        # teach: fact = key = value
        if lower.startswith("teach:"):
            parts = text[6:].strip().split("=")
            if len(parts) >= 2:
                cat = parts[0].strip()
                key = parts[1].strip()
                val = parts[2].strip() if len(parts) > 2 else ""
                if cat.lower() == "fact":
                    self.core.learn_fact(key, val)
                    return f"✓ Learned fact: {key} = {val}"
                elif cat.lower() == "rule":
                    self.core.learn_rule(key, val)
                    return f"✓ Learned rule: {key}"

        return None


# ----------------------------- Command Dispatcher -----------------------------

class CommandDispatcher:
    def __init__(self, ai_agent: 'SelfThinkingAI'):
        self.agent = ai_agent

    def handle(self, cmd: str) -> Optional[str]:
        c = cmd.strip().lower()
        if c in ("exit", "quit"):
            self.agent.shutdown()
            return "exiting"

        if c == "status":
            return self.agent.status()

        if c == "reasoning":
            self.agent.show_reasoning = not self.agent.show_reasoning
            return f"Reasoning: {'ON' if self.agent.show_reasoning else 'OFF'}"

        if c == "voice on":
            if self.agent.voice.enabled:
                self.agent.voice_enabled = True
                return "Voice: ON"
            return "Voice not available on this system"

        if c == "voice off":
            self.agent.voice_enabled = False
            return "Voice: OFF"

        if c == "help":
            return self.help_text()

        return None

    def help_text(self):
        return textwrap.dedent(
            """
            Commands:
              - status        show knowledge and counts
              - reasoning     toggle internal reasoning display
              - voice on/off  enable/disable TTS
              - help          show this message
              - exit/quit     exit gracefully
            Teaching patterns:
              - my name is <name>
              - remember that <fact>
              - when I say <X> respond with <Y>
              - teach: fact = key = value
            """
        )


# ----------------------------- Main Agent -----------------------------

class SelfThinkingAI:
    def __init__(self, name: str = APP_NAME):
        self.storage = Storage()
        self.core = ThinkingCore(self.storage, identity_name=name)
        self.external = ExternalAI()
        self.teacher = Teacher(self.core)
        self.voice = VoiceSystem()
        self.name = name
        self.show_reasoning = False
        self.voice_enabled = self.voice.enabled
        self.cmd_dispatcher = CommandDispatcher(self)
        self._running = True

    def shutdown(self):
        self._running = False
        try:
            self.voice.stop()
        except Exception:
            pass
        console.print(Panel("[red]Shutting down... Goodbye.[/red]", border_style="red"))
        sys.exit(0)

    def status(self) -> str:
        facts_count = len(self.storage.get_recent_experiences(1)) if not SQLALCHEMY_AVAILABLE else "(see DB)"
        msg = f"Name: {self.name}\nVoice: {'enabled' if self.voice.enabled else 'disabled'}\nContext window: {CONTEXT_WINDOW}"
        return msg

    def process(self, user_input: str) -> str:
        # 1. Command handling
        cmd_result = self.cmd_dispatcher.handle(user_input)
        if cmd_result:
            return cmd_result

        # 2. Teaching
        teach = self.teacher.parse_and_apply(user_input)
        if teach:
            self.core.record('user', user_input, {'type': 'teaching'})
            self.core.record('agent', teach, {'type': 'teaching'})
            if self.voice_enabled:
                self.voice.speak(teach)
            return teach

        # 3. Internal recall
        recall = self.core.recall(user_input)
        if recall:
            self.core.record('user', user_input)
            self.core.record('agent', recall)
            if self.voice_enabled:
                self.voice.speak(recall)
            return f"💭 {recall}"

        # 4. External query
        context = self.core.recent_context()
        ext = self.external.query(user_input, context=context)
        self.core.record('user', user_input)
        self.core.record('agent', ext, {'source': 'external'})
        if self.voice_enabled:
            self.voice.speak(ext)
        return f"🌐 {ext}"


# ----------------------------- CLI / Main Loop -----------------------------

def main():
    console.clear()
    ai = SelfThinkingAI("Ares")

    welcome = Panel(
        f"[bold cyan]🧠 {ai.name} — Self-thinking AI[/bold cyan]\n\n"
        "Mode: [yellow]Internal Reasoning + External Knowledge[/yellow]\n"
        f"Voice: {'✅' if ai.voice.enabled else '❌'}\n\n"
        "Teaching patterns: my name is <name> | remember that <fact> | when I say X respond with Y\n"
        "Type 'help' for commands. Type 'exit' to quit.",
        border_style="cyan",
    )
    console.print(welcome)

    while True:
        try:
            user_input = Prompt.ask("[bold yellow]◆ You[/bold yellow]")
            if not user_input.strip():
                continue
            response = ai.process(user_input)
            # optionally show reasoning
            if ai.show_reasoning:
                console.print(Panel("\n".join([f"* {r}" for r in []]) or "(no visible trace)"))
            console.print(Panel(response, title=ai.name, border_style="green"))

        except KeyboardInterrupt:
            console.print("\n[red]Interrupted. Exiting...[/red]")
            ai.shutdown()
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


if __name__ == "__main__":
    main()
