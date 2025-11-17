### What is htop

`htop` is an interactive process viewer for Linux systems that shows real-time system metrics such as CPU, memory, load average, and running processes. It is more user-friendly than the traditional `top` command and supports colors, scrolling, and interactive actions.

### Key Features of htop

* Real-time visualization of CPU, memory, and swap usage
* Ability to scroll vertically and horizontally to see all processes
* Sort processes by CPU, memory, PID, user and other fields
* Supports interactive actions like killing, renicing, and filtering processes
* Displays tree view to understand parent-child process hierarchy

### How to Install htop

On Debian or Ubuntu

```
sudo apt install htop
```

On CentOS, RHEL or Amazon Linux

```
sudo yum install htop
```

On Fedora

```
sudo dnf install htop
```

### Basic Usage

Run the command

```
htop
```

You will see colored bars for each CPU core, memory, and swap usage followed by a full list of processes.

### Important Sections of the htop Interface

* System metrics: CPU, memory, swap, load average, and uptime
* Process list: All running tasks with details like PID, USER, CPU%, MEM%
* Menu bar: Options like Setup, Search, Sort, Tree, Kill, Renice

### Useful Keyboard Shortcuts

* F2: Setup menu
* F3: Search process
* F4: Filter processes
* F5: Tree view
* F6: Sort by specific column
* F9: Kill a process
* F10: Quit htop

### Common Use Cases

* Identifying processes consuming high CPU or RAM
* Inspecting rogue or zombie processes
* Viewing the parent-child structure of running services
* Monitoring system performance during deployments or load tests

### Difference Between top and htop

* htop shows a more readable, color-coded interface
* htop allows scrolling for complete visibility
* htop supports interactive killing and renicing of processes
* htop provides tree view for better process hierarchy understanding

If you want, I can also create a comparison table or show advanced htop configurations.


### Viewing Real-Time Processes

Run the basic command to see CPU, memory usage, load average, and active processes

```
top
```

### Sorting Processes by CPU or Memory

Inside top, press

* `P` to sort by CPU usage
* `M` to sort by memory usage

### Filtering Processes by User

To show processes of a specific user

```
top -u username
```

### Changing the Refresh Interval

To refresh every 2 seconds

```
top -d 2
```

### Killing a Process from top

Press `k` inside top, enter the PID, then specify the signal (default is SIGTERM).

### Showing Full Command of a Process

Press `c` inside top to toggle full command-line display.

### Displaying Only Processes of a Specific PID

```
top -p <pid>
```

### Running top in Batch Mode

Useful for logging or scripts

```
top -b -n 1
```

If you want, I can also share advanced monitoring tricks using top.
Here is the **answer to the earlier medium-level Linux/DevOps question** about investigating a CPU spike with multiple Java processes:

### Investigating High CPU Usage on a Linux Server

When CPU spikes to 90 percent, start with `htop` or `top` to identify which Java processes are consuming the most CPU. Use `ps -p <pid> -o pid,user,pcpu,pmem,cmd` to inspect the exact command that launched each process. Run `lsof -p <pid>` to check open files, sockets, or JARs associated with the process, and explore `/proc/<pid>`, especially `status`, `cmdline`, and `stack`, for thread and memory details. Finally, correlate this information with system logs in `/var/log`, service logs (like systemd journal), and recent deployment or configuration changes to identify whether the spike is caused by a code change, runaway thread, memory leak, or external load.
