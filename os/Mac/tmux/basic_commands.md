### **tmux Basic Commands for macOS**  

tmux is a terminal multiplexer that allows you to create, manage, and switch between multiple terminal sessions.

---

### **1️⃣ Starting & Exiting tmux**
| Command | Description |
|---------|------------|
| `tmux` | Start a new tmux session. |
| `tmux new -s mysession` | Start a new session named `mysession`. |
| `tmux ls` | List all running tmux sessions. |
| `tmux attach -t mysession` | Attach to an existing session. |
| `tmux detach` (Ctrl + B, then D) | Detach from the current session. |
| `tmux kill-session -t mysession` | Kill a specific session. |
| `tmux kill-server` | Kill all tmux sessions. |

---

### **2️⃣ Pane Management**
| Command | Description |
|---------|------------|
| `Ctrl + B, %` | Split window **vertically**. |
| `Ctrl + B, "` | Split window **horizontally**. |
| `Ctrl + B, Arrow Key` | Move between panes. |
| `Ctrl + B, Z` | Zoom into a pane (toggle). |
| `Ctrl + B, X` | Close the current pane. |

---

### **3️⃣ Window Management**
| Command | Description |
|---------|------------|
| `Ctrl + B, C` | Create a new window. |
| `Ctrl + B, N` | Next window. |
| `Ctrl + B, P` | Previous window. |
| `Ctrl + B, W` | List all windows. |
| `Ctrl + B, &` | Close the current window. |

---

### **4️⃣ Session Management**
| Command | Description |
|---------|------------|
| `Ctrl + B, D` | Detach from the session. |
| `tmux attach -t mysession` | Reattach to a session. |
| `tmux rename-session -t oldname newname` | Rename a session. |

---

### **5️⃣ Miscellaneous**
| Command | Description |
|---------|------------|
| `Ctrl + B, [` | Enter scroll mode. |
| `Ctrl + B, ]` | Paste copied text. |
| `Ctrl + B, Space` | Cycle through layouts. |
| `tmux set-option -g mouse on` | Enable mouse support. |

Let me know if you need more details! 🚀