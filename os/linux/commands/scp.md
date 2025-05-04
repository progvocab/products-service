The **`scp` (Secure Copy Protocol)** command is used to securely transfer files between local and remote systems (or between two remote systems) over SSH. Below is a step-by-step guide to copy files from a **Mac (local)** to a **Linux (remote)** machine.

---

### **1. Basic Syntax of `scp`**
```bash
scp [options] <source_file> <username>@<remote_host>:<destination_path>
```
- `source_file`: File or directory to copy (on your Mac).
- `username`: Remote Linux user.
- `remote_host`: IP address or domain of the Linux machine.
- `destination_path`: Target directory on the Linux machine.

---

### **2. Copy a Single File (Mac → Linux)**
```bash
scp /path/to/local/file.txt username@linux-server-ip:/path/to/remote/directory/
```
#### **Example**:
```bash
scp ~/Documents/report.txt alice@192.168.1.100:/home/alice/files/
```
- Copies `report.txt` from your Mac to the Linux machine's `/home/alice/files/` directory.

---

### **3. Copy a Directory (Recursively)**
Use `-r` to copy directories recursively:
```bash
scp -r ~/Desktop/my_project alice@192.168.1.100:/home/alice/projects/
```

---

### **4. Specify a Different SSH Port**
If the remote Linux server uses a non-default SSH port (not 22), use `-P`:
```bash
scp -P 2222 ~/file.txt alice@192.168.1.100:/home/alice/
```

---

### **5. Preserve File Attributes**
Use `-p` to preserve modification times and permissions:
```bash
scp -p ~/file.txt alice@192.168.1.100:/home/alice/
```

---

### **6. Verbose Mode (Debugging)**
Add `-v` to see detailed transfer logs:
```bash
scp -v ~/file.txt alice@192.168.1.100:/home/alice/
```

---

### **7. Key-Based Authentication (Passwordless)**
To avoid entering passwords repeatedly:
1. **Generate SSH keys** on your Mac (if not done already):
   ```bash
   ssh-keygen -t ed25519
   ```
2. **Copy the public key** to the Linux server:
   ```bash
   ssh-copy-id alice@192.168.1.100
   ```
3. Now `scp` will work without passwords:
   ```bash
   scp ~/file.txt alice@192.168.1.100:/home/alice/
   ```

---

### **8. Common Errors & Fixes**
| Error                          | Solution                                                                 |
|--------------------------------|--------------------------------------------------------------------------|
| **"Permission denied"**        | Ensure the remote user has write access to the destination directory.    |
| **"No such file or directory"**| Verify paths exist on both systems.                                      |
| **Connection refused**         | Check if SSH is running on the Linux server (`sudo systemctl status sshd`). |

---

### **Example Workflow**
1. **On your Mac**, open **Terminal**.
2. Run:
   ```bash
   scp ~/Downloads/data.csv bob@203.0.113.45:/home/bob/backups/
   ```
3. Enter the Linux user's password when prompted.

---

### **Key Takeaways**
- `scp` uses **SSH encryption** for secure transfers.
- Use `-r` for directories, `-P` for custom ports, and `-p` to preserve metadata.
- For frequent transfers, set up **SSH key authentication**.

Need to copy from Linux → Mac instead? Just reverse the arguments:
```bash
scp alice@192.168.1.100:/remote/file.txt ~/local/directory/
```