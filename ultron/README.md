# ULTRON

A personal AI that lives inside your computer and acts as your second pair of hands.

You talk to it, it hears you, it answers out loud — and it can actually *do things*:
run commands, read and write files, open apps and websites, control the keyboard and
mouse, see your screen, and search the web. It remembers you between sessions and
teaches itself new skills as it goes. It's driven by **Claude Opus 5.5** and wrapped in
a glowing, voice-reactive HUD.

![the core reacts to your voice and to what it's doing]

---

## What it can do

- **Hear you** — the HUD listens through your browser's speech recognition.
- **Talk back** — it speaks its replies in a calm, low machine voice.
- **Run your computer** — shell commands, Python, files, apps, URLs, keyboard, mouse, screenshots.
- **Search the web** — built-in, via Claude's server-side web search.
- **Remember** — durable facts (`remember` / `recall`) stored in `data/memory.json`.
- **Learn skills by itself** — when it works out how to do something repeatable, it saves a
  Python skill (`data/skills/`) and reuses it next time.
- **Stay safe** — anything destructive (shell, python, writing/deleting files, running a skill)
  pops an **Authorize** card in the HUD before it runs. Flip on `god_mode` to remove the gate.

---

## Setup

Requires **Python 3.10+**.

```bash
cd ultron
python -m venv venv
# Windows:  venv\Scripts\activate      macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt

# Give it your Anthropic API key (get one at console.anthropic.com):
export ANTHROPIC_API_KEY=sk-ant-...       # Windows: setx ANTHROPIC_API_KEY sk-ant-...

python run.py
```

It opens `http://127.0.0.1:8787` in your browser. **Use Chrome or Edge** for voice input.
Click the glowing core once to wake it (the browser needs that click to grant the microphone),
then just talk. You can also type in the box at the bottom.

> First run with no key? It still launches the HUD so you can look around — it just can't
> think until the key is set. An `ant auth login` profile works instead of the env var.

---

## Using it

- **Click the core** → grants the mic and starts listening.
- **🎙 VOICE** → toggle listening on/off.
- **Type + SEND** → works even without a mic.
- **◼** → interrupt it mid-task.
- When it wants to run something risky, an **ALLOW / DENY** card appears. You decide.

Try: *"what's in my Downloads folder?"*, *"open YouTube"*, *"take a screenshot and tell me what's
on screen"*, *"remember that my main project is at ~/code/app"*, *"search the web for today's weather
in Casablanca and read it to me."*

---

## Configuration

Copy `config.example.json` to `config.json` and edit:

| key             | meaning |
|-----------------|---------|
| `model`         | Claude model (default `claude-opus-5-5`). |
| `effort`        | `low` · `medium` · `high` · `xhigh` · `max` — how hard it thinks. |
| `web_search`    | enable the web search tool. |
| `god_mode`      | `true` removes the approval gate — it runs everything without asking. Use with care. |
| `workdir`       | base folder for file/shell tools (default: your home folder). |
| `voice`         | preferred speech-synthesis voice name, e.g. `"Google UK English Male"`. |
| `port` / `host` | where the server listens. |

---

## How it's built

```
ultron/
├─ run.py              # entry point — starts the server, opens the HUD
├─ config.example.json
├─ requirements.txt
├─ web/
│  └─ hud.html         # the glowing, voice-reactive interface (speech in/out lives here)
├─ ultron/
│  ├─ server.py        # FastAPI + WebSocket; one Brain per browser connection
│  ├─ brain.py         # the Claude Opus 5.5 reasoning + tool-use loop (streaming)
│  ├─ tools.py         # the hands: shell, python, files, apps, keyboard/mouse, screen
│  └─ memory.py        # persistent memory + self-taught skills
└─ data/               # created at runtime: memory.json, skills/
```

The HUD does speech-to-text and text-to-speech in the browser (no heavy local models) and
talks to the Python backend over a WebSocket. The backend runs the agent loop: it streams
Claude's words to the HUD live, executes tools on your machine, and asks you to approve the
dangerous ones.

---

## Safety

This program gives an AI real control of your computer. By design:

- Destructive tools are gated behind your explicit approval unless you enable `god_mode`.
- It binds to `127.0.0.1` only — nobody else on the network can reach it.
- Run it as **yourself**, on **your own machine**, and read the Authorize cards before clicking ALLOW.

Treat `god_mode` the way you'd treat handing someone your unlocked laptop.
