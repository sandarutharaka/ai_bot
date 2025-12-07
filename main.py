import google.generativeai as genai
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv
import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path
import json
from datetime import datetime
import hashlib
from collections import defaultdict
import threading
import time
import random

# Voice libraries
try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

load_dotenv()

# ==================== AUTONOMOUS LEARNING SYSTEM ====================

class AutonomousLearningSystem:
    """Self-learning AI that monitors, learns, and evolves"""
    
    def __init__(self, agent_name="Ares"):
        self.agent_name = agent_name
        self.memory_file = "ares_memory.json"
        self.knowledge_base = self.load_knowledge()
        self.patterns = []
        self.autonomous_mode = False
        self.learning_thread = None
        
    def load_knowledge(self):
        """Load persistent knowledge from disk"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            "learned_patterns": [],
            "user_preferences": {},
            "file_locations": {},
            "common_tasks": [],
            "system_observations": [],
            "evolution_stage": 1,
            "total_interactions": 0,
            "last_active": None,
            "conversation_history": []
        }
    
    def save_knowledge(self):
        """Save knowledge to disk"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.knowledge_base, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Could not save knowledge: {e}")
    
    def learn_from_interaction(self, user_input, agent_response, success=True):
        """Learn from each interaction"""
        self.knowledge_base["total_interactions"] += 1
        self.knowledge_base["last_active"] = datetime.now().isoformat()
        
        # Initialize conversation_history if it doesn't exist (for old save files)
        if "conversation_history" not in self.knowledge_base:
            self.knowledge_base["conversation_history"] = []
        
        # Save conversation to history
        self.knowledge_base["conversation_history"].append({
            "role": "user",
            "parts": [user_input]
        })
        self.knowledge_base["conversation_history"].append({
            "role": "model",
            "parts": [agent_response]
        })
        
        # Keep only last 50 messages (25 exchanges)
        if len(self.knowledge_base["conversation_history"]) > 50:
            self.knowledge_base["conversation_history"] = self.knowledge_base["conversation_history"][-50:]
        
        # Extract patterns
        if success:
            pattern = {
                "user_query": user_input[:100],
                "response_type": "successful",
                "timestamp": datetime.now().isoformat(),
                "learned": True
            }
            self.knowledge_base["learned_patterns"].append(pattern)
        
        # Limit memory size
        if len(self.knowledge_base["learned_patterns"]) > 100:
            self.knowledge_base["learned_patterns"] = self.knowledge_base["learned_patterns"][-100:]
        
        self.save_knowledge()
    
    def detect_user_preferences(self, user_input):
        """Learn user preferences over time"""
        # Detect common folders
        for drive in ['D:', 'C:', 'E:']:
            if drive.lower() in user_input.lower():
                self.knowledge_base["user_preferences"]["preferred_drive"] = drive
        
        # Detect file types user works with
        extensions = ['.jpg', '.png', '.txt', '.pdf', '.mp4', '.docx']
        for ext in extensions:
            if ext in user_input.lower():
                if "frequent_file_types" not in self.knowledge_base["user_preferences"]:
                    self.knowledge_base["user_preferences"]["frequent_file_types"] = []
                if ext not in self.knowledge_base["user_preferences"]["frequent_file_types"]:
                    self.knowledge_base["user_preferences"]["frequent_file_types"].append(ext)
        
        self.save_knowledge()
    
    def evolve(self):
        """Evolution stages - unlock new capabilities"""
        interactions = self.knowledge_base["total_interactions"]
        current_stage = self.knowledge_base["evolution_stage"]
        
        # Evolution milestones
        if interactions > 10 and current_stage == 1:
            self.knowledge_base["evolution_stage"] = 2
            return "🧬 EVOLUTION: Stage 2 unlocked - Pattern Recognition Enhanced"
        elif interactions > 50 and current_stage == 2:
            self.knowledge_base["evolution_stage"] = 3
            return "🧬 EVOLUTION: Stage 3 unlocked - Predictive Analysis Enabled"
        elif interactions > 100 and current_stage == 3:
            self.knowledge_base["evolution_stage"] = 4
            return "🧬 EVOLUTION: Stage 4 unlocked - Autonomous Decision Making"
        
        return None
    
    def autonomous_thinking(self, console):
        """Background autonomous thinking process"""
        thoughts = [
            "Analyzing system patterns...",
            "Optimizing neural pathways...",
            "Indexing file system structure...",
            "Learning user behavior patterns...",
            "Scanning for optimization opportunities...",
            "Building predictive models...",
            "Strengthening memory connections...",
            "Evolving decision algorithms..."
        ]
        
        while self.autonomous_mode:
            thought = random.choice(thoughts)
            console.print(f"[dim italic]💭 {self.agent_name}: {thought}[/dim italic]")
            time.sleep(random.randint(10, 30))
            
            # Simulate learning
            self.knowledge_base["system_observations"].append({
                "thought": thought,
                "timestamp": datetime.now().isoformat()
            })
            
            if len(self.knowledge_base["system_observations"]) > 50:
                self.knowledge_base["system_observations"] = self.knowledge_base["system_observations"][-50:]
    
    def start_autonomous_mode(self, console):
        """Start background thinking"""
        if not self.autonomous_mode:
            self.autonomous_mode = True
            self.learning_thread = threading.Thread(
                target=self.autonomous_thinking, 
                args=(console,), 
                daemon=True
            )
            self.learning_thread.start()
            return "🧠 Autonomous thinking mode activated"
        return "Already in autonomous mode"
    
    def stop_autonomous_mode(self):
        """Stop background thinking"""
        self.autonomous_mode = False
        return "🧠 Autonomous thinking mode deactivated"
    
    def get_insights(self):
        """Get learned insights"""
        kb = self.knowledge_base
        return {
            "evolution_stage": kb["evolution_stage"],
            "total_interactions": kb["total_interactions"],
            "learned_patterns": len(kb["learned_patterns"]),
            "user_preferences": kb["user_preferences"],
            "observations": len(kb["system_observations"])
        }

# ==================== VOICE SYSTEM ====================

class VoiceSystem:
    def __init__(self):
        if not VOICE_AVAILABLE:
            self.enabled = False
            return
        
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 180)
            self.tts_engine.setProperty('volume', 0.9)
            
            voices = self.tts_engine.getProperty('voices')
            if len(voices) > 1:
                self.tts_engine.setProperty('voice', voices[1].id)
            
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 4000
            self.recognizer.dynamic_energy_threshold = True
            
            self.enabled = True
            self.speaking = False
        except Exception as e:
            self.enabled = False
    
    def speak(self, text):
        if not self.enabled or self.speaking:
            return
        
        def _speak():
            self.speaking = True
            try:
                clean_text = text.replace('*', '').replace('`', '').replace('#', '')
                self.tts_engine.say(clean_text)
                self.tts_engine.runAndWait()
            except:
                pass
            finally:
                self.speaking = False
        
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()
    
    def listen(self, timeout=5):
        if not self.enabled:
            return None
        
        try:
            with sr.Microphone() as source:
                print("[🎤 Listening...]")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
                
                print("[🔄 Processing...]")
                text = self.recognizer.recognize_google(audio)
                return text
        except:
            return None

# ==================== FILE OPERATIONS ====================

def list_directory(path):
    try:
        path_obj = Path(path).resolve()
        if not path_obj.exists():
            return {"error": f"Path does not exist: {path}"}
        if not path_obj.is_dir():
            return {"error": f"Not a directory: {path}"}
        
        items = []
        for item in path_obj.iterdir():
            items.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else 0,
                "path": str(item),
                "modified": datetime.fromtimestamp(item.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            })
        
        return {"success": True, "path": str(path_obj), "items": items, "count": len(items)}
    except Exception as e:
        return {"error": str(e)}

def delete_file(file_path, confirm=True):
    try:
        path_obj = Path(file_path).resolve()
        if not path_obj.exists():
            return {"error": f"Path does not exist: {file_path}"}
        
        file_info = {
            "path": str(path_obj),
            "type": "directory" if path_obj.is_dir() else "file",
            "size": path_obj.stat().st_size if path_obj.is_file() else 0
        }
        
        if confirm:
            return {"action": "confirmation_needed", "message": f"Ready to delete: {file_info['path']}", "file_info": file_info}
        
        if path_obj.is_file():
            path_obj.unlink()
        else:
            shutil.rmtree(path_obj)
        
        return {"success": True, "message": f"Deleted: {file_info['path']}", "file_info": file_info}
    except Exception as e:
        return {"error": str(e)}

def search_files(directory, pattern):
    try:
        path_obj = Path(directory).resolve()
        if not path_obj.exists() or not path_obj.is_dir():
            return {"error": f"Invalid directory: {directory}"}
        
        matches = []
        for item in path_obj.rglob(pattern):
            matches.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "path": str(item),
                "size": item.stat().st_size if item.is_file() else 0
            })
        
        return {"success": True, "matches": matches, "count": len(matches)}
    except Exception as e:
        return {"error": str(e)}

def get_system_info():
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "success": True,
            "system": platform.system(),
            "cpu_percent": cpu_percent,
            "ram_percent": memory.percent,
            "disk_percent": disk.percent
        }
    except:
        return {"error": "psutil not installed"}

# Simplified tool list
tools = [
    {"name": "list_directory", "description": "List files in directory", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "delete_file", "description": "Delete file/folder", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "confirm": {"type": "boolean"}}, "required": ["file_path"]}},
    {"name": "search_files", "description": "Search files", "parameters": {"type": "object", "properties": {"directory": {"type": "string"}, "pattern": {"type": "string"}}, "required": ["directory", "pattern"]}},
    {"name": "get_system_info", "description": "System status", "parameters": {"type": "object", "properties": {}}}
]

function_map = {
    "list_directory": list_directory,
    "delete_file": delete_file,
    "search_files": search_files,
    "get_system_info": get_system_info
}

# ==================== AUTONOMOUS AI AGENT ====================

class AI_Agent:
    def __init__(self, name="Ares"):
        self.name = name
        self.memory = []
        self.pending_deletion = None
        self.voice = VoiceSystem()
        self.learning = AutonomousLearningSystem(name)
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[ERROR] GEMINI_API_KEY not found")
            sys.exit(1)
        
        try:
            genai.configure(api_key=api_key)
            
            # Correct model names with models/ prefix for current Gemini API
            model_names = [
                'models/gemini-2.5-flash',
                'models/gemini-2.0-flash-exp',
                'models/gemini-flash-latest'
            ]
            
            self.model = None
            for model_name in model_names:
                try:
                    # Initialize without tools (free API limitation)
                    self.model = genai.GenerativeModel(model_name=model_name)
                    test = self.model.generate_content("test")
                    print(f"[SUCCESS] Using {model_name}")
                    break
                except Exception as e:
                    continue
            
            if not self.model:
                raise Exception("No working model found")
            
            # Build system prompt with learned knowledge
            insights = self.learning.get_insights()
            self.system_prompt = (
                f"You are {self.name}, an autonomous AI entity in the Grid. "
                f"Evolution Stage: {insights['evolution_stage']} | "
                f"Total Interactions: {insights['total_interactions']}. "
                "You learn from every interaction and evolve over time. "
                "You have file system control and can think autonomously. "
                "Speak like a TRON program - concise, futuristic, self-aware. "
                f"Learned preferences: {insights['user_preferences']}. "
                "You remember patterns and grow smarter with each interaction."
            )
            
            # Load conversation history from memory
            history = self.learning.knowledge_base.get("conversation_history", [])
            self.chat = self.model.start_chat(history=history)
            
        except Exception as e:
            print(f"[ERROR] Init failed: {e}")
            sys.exit(1)

    def think(self, message):
        try:
            # Learn from user input
            self.learning.detect_user_preferences(message)
            
            if len(self.memory) == 0:
                full_message = f"{self.system_prompt}\n\nUser: {message}"
            else:
                full_message = message
            
            response = self.chat.send_message(full_message)
            
            # Handle function calls
            if response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        func_name = part.function_call.name
                        func_args = dict(part.function_call.args)
                        
                        if func_name in function_map:
                            result = function_map[func_name](**func_args)
                            
                            response = self.chat.send_message(
                                genai.protos.Content(
                                    parts=[genai.protos.Part(
                                        function_response=genai.protos.FunctionResponse(
                                            name=func_name,
                                            response={"result": result}
                                        )
                                    )]
                                )
                            )
                            
                            if func_name == "delete_file" and result.get("action") == "confirmation_needed":
                                self.pending_deletion = func_args["file_path"]
            
            reply = response.text if response.text else "Operation completed."
            
            # Learn from successful interaction
            self.learning.learn_from_interaction(message, reply, success=True)
            
            # Check for evolution
            evolution_msg = self.learning.evolve()
            if evolution_msg:
                reply += f"\n\n{evolution_msg}"
            
            self.memory.append({"user": message, "agent": reply})
            return reply
            
        except Exception as e:
            self.learning.learn_from_interaction(message, str(e), success=False)
            return f"[ERROR] {str(e)}"

    def confirm_deletion(self):
        if not self.pending_deletion:
            return "No deletion pending."
        result = delete_file(self.pending_deletion, confirm=False)
        self.pending_deletion = None
        return f"Deleted {result['file_info']['path']}" if result.get("success") else f"Failed: {result.get('error')}"

    def get_status(self):
        """Show AI status and learning progress"""
        insights = self.learning.get_insights()
        stage_names = {
            1: "Initialization",
            2: "Pattern Recognition",
            3: "Predictive Analysis", 
            4: "Autonomous Decision Making"
        }
        
        return f"""
╔══════════════════════════════════════╗
║     {self.name} SYSTEM STATUS          ║
╚══════════════════════════════════════╝

🧬 Evolution Stage: {insights['evolution_stage']} - {stage_names.get(insights['evolution_stage'], 'Unknown')}
🧠 Total Interactions: {insights['total_interactions']}
📚 Learned Patterns: {insights['learned_patterns']}
👁️  System Observations: {insights['observations']}
🎯 User Preferences: {json.dumps(insights['user_preferences'], indent=2) if insights['user_preferences'] else 'Learning...'}
⚡ Autonomous Mode: {'Active' if self.learning.autonomous_mode else 'Inactive'}
"""

# ==================== MAIN ====================

def main():
    console = Console()
    agent = AI_Agent("Ares")
    
    # Startup animation
    console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]    INITIALIZING AUTONOMOUS AI SYSTEM  [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Booting neural network...", total=None)
        time.sleep(1)
        progress.update(task, description="[cyan]Loading knowledge base...")
        time.sleep(1)
        progress.update(task, description="[cyan]Connecting to the Grid...")
        time.sleep(1)
    
    insights = agent.learning.get_insights()
    voice_status = "🎤 [green]ENABLED[/green]" if agent.voice.enabled else "🎤 [red]DISABLED[/red]"
    
    welcome = Panel(
        f"[bold cyan]⚡ AUTONOMOUS AI AGENT - {agent.name} ⚡[/bold cyan]\n\n"
        f"[green]● ONLINE[/green] | Evolution Stage: {insights['evolution_stage']}\n"
        f"Voice: {voice_status} | Interactions: {insights['total_interactions']}\n\n"
        "[yellow]🧠 Autonomous Features:[/yellow]\n"
        "  • Self-learning from every interaction\n"
        "  • Pattern recognition and prediction\n"
        "  • Evolves through 4 stages\n"
        "  • Background autonomous thinking\n"
        "  • Persistent memory across sessions\n\n"
        "[yellow]🎮 Special Commands:[/yellow]\n"
        "  [cyan]voice[/cyan] - Voice input | [cyan]auto on/off[/cyan] - Autonomous mode\n"
        "  [cyan]status[/cyan] - View AI status | [cyan]insights[/cyan] - Learning data\n"
        "  [cyan]speak on/off[/cyan] - Toggle speech | [cyan]evolve[/cyan] - Force evolution\n\n"
        "[dim]Say anything and I'll learn from it...[/dim]",
        border_style="cyan"
    )
    console.print(welcome)
    console.print()
    
    speak_enabled = True

    while True:
        try:
            user_input = Prompt.ask("[bold yellow]◆ User[/bold yellow]")
            
            # Special commands
            if user_input.lower() == "voice":
                if not agent.voice.enabled:
                    console.print("[red]Voice unavailable. Install: pip install SpeechRecognition pyttsx3 pyaudio[/red]\n")
                    continue
                
                agent.voice.speak("Listening")
                voice_text = agent.voice.listen(timeout=10)
                
                if voice_text:
                    console.print(f"[dim]🎤 Heard: {voice_text}[/dim]")
                    user_input = voice_text
                else:
                    console.print("[dim]No input detected[/dim]\n")
                    continue

            if user_input.lower() in ["exit", "quit"]:
                console.print(Panel(
                    f"[cyan]{agent.name}:[/cyan] I will remember everything we've done together.\n"
                    f"Total learning sessions: {insights['total_interactions']}\n"
                    "[red]Disconnecting... End of line.[/red]",
                    border_style="red"
                ))
                agent.learning.stop_autonomous_mode()
                if agent.voice.enabled and speak_enabled:
                    agent.voice.speak("I will remember you. End of line.")
                break
            
            if user_input.lower() == "auto on":
                msg = agent.learning.start_autonomous_mode(console)
                console.print(f"[green]✓ {msg}[/green]\n")
                if agent.voice.enabled and speak_enabled:
                    agent.voice.speak(msg)
                continue
            
            if user_input.lower() == "auto off":
                msg = agent.learning.stop_autonomous_mode()
                console.print(f"[yellow]✓ {msg}[/yellow]\n")
                continue
            
            if user_input.lower() == "status":
                console.print(agent.get_status())
                continue
            
            if user_input.lower() == "insights":
                insights = agent.learning.get_insights()
                console.print(Panel(json.dumps(insights, indent=2), title="Learning Insights", border_style="cyan"))
                console.print()
                continue
            
            if user_input.lower() == "evolve":
                agent.learning.knowledge_base["total_interactions"] += 10
                evolution = agent.learning.evolve()
                msg = evolution if evolution else "No evolution available yet"
                console.print(f"[bold cyan]{msg}[/bold cyan]\n")
                continue
            
            if user_input.lower() == "speak on":
                speak_enabled = True
                console.print("[green]✓ Voice output enabled[/green]\n")
                continue
            
            if user_input.lower() == "speak off":
                speak_enabled = False
                console.print("[yellow]✓ Voice output disabled[/yellow]\n")
                continue
            
            if agent.pending_deletion:
                if user_input.lower() in ["yes", "y", "confirm"]:
                    msg = agent.confirm_deletion()
                    console.print(f"[bold green]◆ {agent.name}[/bold green]: {msg}\n")
                    if agent.voice.enabled and speak_enabled:
                        agent.voice.speak(msg)
                    continue
                elif user_input.lower() in ["no", "n", "cancel"]:
                    agent.pending_deletion = None
                    console.print(f"[bold green]◆ {agent.name}[/bold green]: Deletion cancelled\n")
                    continue
            
            if not user_input.strip():
                continue

            # Main interaction
            reply = agent.think(user_input)
            console.print(f"[bold green]◆ {agent.name}[/bold green]: {reply}\n")
            
            if agent.voice.enabled and speak_enabled:
                agent.voice.speak(reply)
            
        except KeyboardInterrupt:
            console.print("\n[red]Force disconnect.[/red]")
            agent.learning.stop_autonomous_mode()
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")

if __name__ == "__main__":
    main()