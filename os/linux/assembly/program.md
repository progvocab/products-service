Perfect! Here's a **step-by-step example** of writing, compiling, and debugging an **x86-64 assembly program** on **Linux**, using **registers and observing the program counter (RIP)**.

---

## 🧩 1. 📝 **Assembly Program**: `register_demo.asm`

This simple program:

* Moves values into registers (`RAX`, `RBX`)
* Adds them
* Exits the program using a Linux syscall

```nasm
; register_demo.asm (x86-64 NASM syntax)
section .text
    global _start

_start:
    mov     rax, 5          ; rax = 5
    mov     rbx, 10         ; rbx = 10
    add     rax, rbx        ; rax = rax + rbx = 15

    ; exit system call
    mov     rdi, rax        ; exit code = rax = 15
    mov     rax, 60         ; syscall number for exit
    syscall
```

---

## 🔧 2. **Assemble and Link**

```bash
nasm -f elf64 register_demo.asm -o register_demo.o
ld -o register_demo register_demo.o
```

---

## 🧪 3. **Debug in GDB to Inspect Registers & Program Counter**

```bash
gdb ./register_demo
```

### Inside GDB:

```gdb
start            # begin execution
info registers   # view values of RAX, RBX, RIP (program counter)
si               # step through one instruction at a time
x/i $rip         # disassemble the instruction at the current RIP
```

Repeat `si` to step, and use `info registers` after each step to see how:

* `RAX`, `RBX` change
* `RIP` (program counter) advances to the next instruction

---

## 🔍 Sample GDB Output (after `start`):

```text
RAX: 0x0
RBX: 0x0
RIP: 0x401000      ; address of current instruction (_start)
```

After a few `si` steps:

```text
RAX: 0x5
RBX: 0xa
RIP: 0x401009      ; moved forward
```

---

## 🧠 What You Just Saw

| Element          | Meaning                                                |
| ---------------- | ------------------------------------------------------ |
| `RAX`, `RBX`     | General-purpose registers                              |
| `RIP`            | **Program Counter**: holds address of next instruction |
| `mov`, `add`     | Instructions modifying registers                       |
| `syscall`        | Triggers Linux system call (here, exit)                |
| `info registers` | Shows full CPU register state                          |
| `x/i $rip`       | Shows the assembly at current PC                       |

---

## 🧰 Optional: Disassemble with `objdump`

```bash
objdump -d ./register_demo
```

---

## ✅ Summary

| Tool | Use                                      |
| ---- | ---------------------------------------- |
| NASM | Assemble `.asm` to `.o`                  |
| `ld` | Link object to binary                    |
| GDB  | Debug: view registers, step, inspect PC  |
| RIP  | Shows which instruction is next (the PC) |

---

Let me know if you'd like:

* A 32-bit version using `EAX`, `EBX`, `EIP`
* ARM architecture example on Raspberry Pi / QEMU
* Assembly with loops or conditional jumps using flags and PC

Happy hacking at the register level! 🧠💻
