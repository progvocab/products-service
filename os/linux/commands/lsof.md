### What is lsof

`lsof` stands for List Open Files. It shows all files opened by processes on a Linux system, including regular files, directories, network sockets, pipes, and devices. Since everything in Linux is treated as a file, `lsof` is a powerful troubleshooting tool.

### Show All Open Files

```
lsof
```

Lists every open file by every process. Useful when diagnosing system-wide file usage.

### Find Which Process Is Using a File

```
lsof /var/log/syslog
```

Shows which process has this file open. Helpful when a file cannot be deleted because a process is still holding it.

### Check Which Process Is Using a Port

```
lsof -i :8080
```

Displays the process listening or connected to port 8080. Commonly used to debug web service port conflicts.

### List Network Connections Only

```
lsof -i
```

Shows all active network connections and listening ports.

### List Open Files by a Specific User

```
lsof -u root
```

Displays all files opened by the user root.

### List All Open Files of a Process

```
lsof -p 1234
```

Shows every file (logs, JARs, sockets) opened by process ID 1234.

### Find Deleted but Still Open Files

```
lsof | grep deleted
```

Helps identify processes holding large deleted files that still consume disk space.

### Kill Processes Using a File or Port

To kill a process using port 3306

```
kill -9 $(lsof -t -i :3306)
```

The `-t` option returns only PIDs, making it easy to script.

If you want, I can also explain advanced `lsof` filtering or real troubleshooting use cases.
