Perfect 👇
Here’s a **Mermaid diagram** showing how **virtual and physical memory interact in Linux** — including the **MMU, TLB, page tables, and swap**.

---

```mermaid
flowchart TD
    subgraph USERSPACE["User Space (Process Memory)"]
        A1["Virtual Address Space<br>(Process A)"]
        A2["Virtual Address Space<br>(Process B)"]
        A1 -->|Virtual Address (e.g., 0x7ffd1234)| MMU
        A2 -->|Virtual Address (e.g., 0x7ffe2345)| MMU
    end

    subgraph KERNELSPACE["Kernel Space"]
        PT["Page Tables<br>(Virtual → Physical mappings)"]
        PFH["Page Fault Handler<br>(Loads pages from disk)"]
        SWAP["Swap Space<br>(Disk area for evicted pages)"]
        MMU -->|Lookup| TLB["TLB (Translation Lookaside Buffer)"]
        TLB -->|Cache Miss| PT
        PT -->|Map to Frame| MMU
        PT -->|If page missing| PFH
        PFH -->|Load Page| RAM
        PFH --> SWAP
    end

    subgraph HARDWARE["Hardware (Physical Memory)"]
        RAM["Physical Memory (RAM Frames)"]
        CPU["CPU"]
        DISK["Storage Device (HDD/SSD)"]
    end

    MMU -->|Translated Physical Address| RAM
    RAM -->|Data| CPU
    SWAP -->|Page In/Out| DISK

    classDef user fill:#e3f2fd,stroke:#1565c0,stroke-width:1px,color:#000;
    classDef kernel fill:#ede7f6,stroke:#4527a0,stroke-width:1px,color:#000;
    classDef hw fill:#fbe9e7,stroke:#bf360c,stroke-width:1px,color:#000;

    class USERSPACE user;
    class KERNELSPACE kernel;
    class HARDWARE hw;
```

---

### 🧭 Explanation of the Flow

1. A **process** (in user space) uses a **virtual address** like `0x7ffd1234`.
2. The **MMU (Memory Management Unit)** translates that virtual address → **physical frame** using the **page tables**.
3. If the mapping is already cached in the **TLB**, translation happens instantly.
4. If not, the **kernel** consults the **page tables**.
5. If the page is missing (not in RAM), a **page fault** occurs → handled by the **page fault handler**.
6. The handler loads the missing page from **swap** or a **file on disk** into RAM.
7. Translation is cached again in the **TLB** for faster future access.
8. The **CPU** reads/writes the actual data via the physical memory address.

---

Would you like me to extend this diagram to show **Copy-on-Write (COW)** behavior — i.e., how Linux shares physical pages between parent and child after a `fork()`?
