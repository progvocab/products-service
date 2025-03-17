To enable SSH on Windows 10 with Windows Subsystem for Linux (WSL), you'll need to set up an SSH server inside your WSL distribution (e.g., Ubuntu). WSL itself doesn't come with an SSH server running by default, so you'll install and configure one manually. Here's a step-by-step guide:

### Step 1: Ensure WSL is Installed
1. **Check WSL Installation**:
   - Open PowerShell or Command Prompt as Administrator.
   - Run: `wsl --list` or `wsl -l` to see if a Linux distribution is installed.
   - If not installed, enable WSL and install a distribution (e.g., Ubuntu):
     - Run: `wsl --install` (requires Windows 10 version 2004 or later).
     - Install a distro from the Microsoft Store (e.g., Ubuntu), then launch it to complete the initial setup.

2. **Set Default Distro** (optional):
   - If you have multiple distros, set your preferred one: `wsl --set-default <DistroName>` (e.g., `wsl --set-default Ubuntu`).

### Step 2: Install OpenSSH Server in WSL
1. **Launch WSL**:
   - Open your WSL terminal (e.g., type `wsl` in PowerShell or open the Ubuntu app).

2. **Update Package List**:
   - Run: `sudo apt update` to ensure your package list is current.

3. **Install OpenSSH Server**:
   - Run: `sudo apt install openssh-server` to install the SSH server.

### Step 3: Configure SSH Server
1. **Edit SSH Configuration** (optional but recommended):
   - Open the SSH config file: `sudo nano /etc/ssh/sshd_config`
   - Common changes:
     - Change the port (e.g., `Port 2222`) to avoid conflicts with Windows' SSH if it's running on port 22.
     - Ensure `PasswordAuthentication yes` is uncommented if you want password login (default is usually `yes`).
     - For key-based authentication, ensure `PubkeyAuthentication yes` is uncommented.
   - Save (Ctrl+O, Enter, Ctrl+X in nano).

2. **Start the SSH Server**:
   - Run: `sudo service ssh start`
   - Verify it’s running: `sudo service ssh status` (you should see it’s active).

### Step 4: Adjust Windows Firewall
1. **Allow Incoming Connections**:
   - WSL uses localhost forwarding, so you may not need this for local access, but for external access:
     - Open Windows Defender Firewall settings: Search "firewall" in the Start menu, select "Windows Defender Firewall with Advanced Security."
     - Create a new inbound rule:
       - Rule Type: Port
       - Protocol: TCP
       - Specific Port: 22 (or your custom port, e.g., 2222)
       - Action: Allow the connection
       - Profile: All (Domain, Private, Public)
       - Name: "WSL SSH"

### Step 5: Test SSH Locally
1. **Get WSL IP Address** (optional for external access):
   - In WSL, run: `ip addr show eth0 | grep inet` and note the IP (e.g., `172.17.x.x`).
   - Note: WSL2 IP changes on reboot; for localhost, you can skip this.

2. **Test from Windows**:
   - Open PowerShell or Command Prompt.
   - Run: `ssh <username>@localhost` (replace `<username>` with your WSL username).
   - If you changed the port: `ssh -p 2222 <username>@localhost`.
   - Enter your WSL password when prompted.

### Step 6: Enable SSH on Boot (Optional)
WSL doesn’t use `systemd` by default, so the SSH server won’t start automatically. Here’s a workaround:
1. **Create a Startup Script in WSL**:
   - Run: `echo "sudo service ssh start" > ~/start-ssh.sh`
   - Make it executable: `chmod +x ~/start-ssh.sh`

2. **Run from Windows on Startup**:
   - Open Task Scheduler in Windows:
     - Search "Task Scheduler" in the Start menu.
     - Action > Create Task:
       - General tab: Name it "WSL SSH Start," check "Run with highest privileges."
       - Triggers tab: New > "At log on" > OK.
       - Actions tab: New > Action: "Start a program," Program: `wsl.exe`, Arguments: `-u root /home/<username>/start-ssh.sh` (replace `<username>`).
       - Conditions tab: Uncheck "Start only if on AC power" if needed.
       - OK, enter your Windows credentials if prompted.

### Step 7: Access from Another Machine (Optional)
- For external access, WSL2’s NAT’d network requires port forwarding:
  - In PowerShell (Admin):
    - Find WSL IP: `wsl hostname -I`
    - Forward port: `netsh interface portproxy add v4tov4 listenport=2222 listenaddress=0.0.0.0 connectport=2222 connectaddress=<WSL-IP>`
    - Update firewall rule for external access if needed.
  - Connect from another machine: `ssh -p 2222 <username>@<Windows-IP>`

### Troubleshooting
- **Port Conflict**: If port 22 is taken (e.g., by Windows OpenSSH), use a different port (e.g., 2222).
- **Connection Refused**: Ensure the SSH server is running and the firewall allows the port.
- **IP Changes**: WSL2 IP is dynamic; consider a script to update forwarding or use localhost for simplicity.

Now you should have SSH enabled and running in WSL on Windows 10! Let me know if you need help with any step.