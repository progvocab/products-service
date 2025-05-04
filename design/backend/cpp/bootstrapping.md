**Bootstrapping** in the context of programming refers to the process of **writing a compiler (or other software) for a programming language in that same language**, enabling the language to compile itself. It’s a self-sustaining technique where an initial, minimal version of a compiler is used to build progressively more complex versions, ultimately producing a full-fledged compiler. The term originates from the phrase "pulling oneself up by one’s bootstraps," reflecting the self-reliant nature of the process.

Since you’ve asked about the architecture of C, the `#if` directive, and other C++-related topics (e.g., `std::vector`, `{fmt}`), I’ll explain bootstrapping, detail how it was used to develop the C language and its compiler, and tie it to your programming context. I’ll also address your question about the language in which C was written and provide relevant insights from the C development history. The explanation will be concise, practical, and include examples where applicable.

---

### What is Bootstrapping?

Bootstrapping a compiler involves the following steps:
1. **Initial Compiler**: Write a minimal compiler for a language (or a subset of it) in another language (e.g., assembly, another high-level language, or an earlier language).
2. **Self-Hosting**: Use the initial compiler to compile a more complete version of the compiler written in the target language.
3. **Iterative Improvement**: Repeat the process, enhancing the compiler with each iteration until it supports the full language specification.

**Key Characteristics**:
- **Self-Sufficiency**: Once bootstrapped, the compiler can compile its own source code without relying on external compilers.
- **Portability**: Bootstrapping enables a language to be implemented on new platforms by compiling the compiler’s source code for the target architecture.
- **Challenges**: Requires an initial compiler (often written in a different language), careful handling of language subsets, and ensuring compatibility across iterations.

**Why Bootstrapping?**
- Demonstrates the language’s maturity and usability.
- Reduces dependency on other languages or tools.
- Facilitates maintenance, as the compiler is written in the same language it compiles.
- Enables portability across different systems.

---

### How Bootstrapping Was Used to Develop the C Language and Compiler

The C programming language, developed by **Dennis Ritchie** at **Bell Labs** in the early 1970s, was created to support system programming for the Unix operating system. The development of C and its compiler heavily relied on bootstrapping, evolving from earlier languages (B and BCPL) and leveraging Unix’s portable design. Below is a detailed account of how bootstrapping was applied, based on historical context and your question about C’s implementation.

#### 1. **Pre-C Context: B and BCPL**
- **Background**: Before C, Ken Thompson developed **B**, a simplified version of **BCPL** (Basic Combined Programming Language), for early Unix development on the PDP-7 and PDP-11 computers (circa 1970).
- **B Compiler**: The B compiler was initially written in **assembly language** or **TMG** (a macro processor) for the PDP-7. Later, it was rewritten in B itself for the PDP-11, making B self-hosting (an early example of bootstrapping).
- **Limitations of B**:
  - B was typeless (everything was a “word”), lacking support for structured data or modern hardware features like byte-addressable memory.
  - It was inefficient on the PDP-11, which had richer addressing modes and data types.
- **Need for C**: To address B’s limitations, Dennis Ritchie began developing C (initially called “NB” for “New B”) around 1971–1973, adding types (`int`, `char`, `struct`) and other features.

#### 2. **Initial C Compiler**
- **First Step (1971–1972)**: Ritchie wrote an early C compiler in **assembly language** or **B** for the PDP-11, targeting a subset of the C language. This compiler was minimal, supporting basic features like functions, basic types, and pointers.
- **Language Subset**: The initial C was not fully featured. It lacked some modern C constructs (e.g., `struct` was added later) but was sufficient to compile simple programs and parts of Unix.
- **Role of Unix**: The Unix kernel, originally written in assembly and B, was gradually rewritten in C. The C compiler was used to compile Unix utilities and kernel components, driving its development.

#### 3. **Bootstrapping the C Compiler**
- **Step 1: Cross-Compilation** (1972–1973):
  - The initial C compiler (written in assembly or B) compiled an early version of the C compiler written in **C itself** (a subset of C).
  - This produced a new compiler binary, still limited but now written in C.
  - Example process:
    - **Compiler A** (in assembly/B): Compiles `compiler.c` (written in C subset).
    - Output: **Compiler B** (binary, runs on PDP-11, understands C subset).
- **Step 2: Self-Hosting**:
  - The new C compiler (Compiler B) was used to compile an improved version of itself, also written in C, adding more features (e.g., `struct`, better pointer arithmetic).
  - This process was iterative:
    - **Compiler B** compiles `compiler_v2.c` (improved C source).
    - Output: **Compiler C** (more complete, still in C).
  - Each iteration added features, making the compiler more capable and closer to the full C language.
- **Step 3: Full C Compiler** (by 1973–1975):
  - By 1973, the C compiler was fully self-hosting, meaning it could compile its own source code entirely in C without relying on B or assembly.
  - The compiler supported most of what became **K&R C** (defined in *The C Programming Language* by Kernighan and Ritchie, 1978).
  - The Unix kernel was largely rewritten in C by this time, proving C’s viability.

#### 4. **Language C Was Written In**
- As you asked in your earlier question, C itself is not “written” in a language, as it’s defined by a **standard** (e.g., ANSI C89, C99). However, the **C compiler** was developed as follows:
  - **Initial Compiler**: Written in **assembly** (for PDP-7/PDP-11) or **B** to bootstrap the process.
  - **Subsequent Compilers**: Written in **C**, with each version compiled by the previous one.
  - **Standard Library**: Libraries like `libc` (e.g., `stdio.h`, `stdlib.h`) were written in **C**, with some assembly for low-level operations (e.g., system calls).
  - **Modern Compilers**: Today’s C compilers (e.g., GCC, Clang) are written in **C** and **C++**, with assembly for architecture-specific optimizations.

#### 5. **Bootstrapping Challenges in C**
- **Subset Limitation**: Early C compilers supported only a subset of C, requiring careful design to ensure the compiler’s source code was compatible with the subset.
- **Portability**: To port C to new architectures (e.g., from PDP-11 to VAX), the compiler was cross-compiled on an existing platform to produce a binary for the target platform.
- **Testing**: Each bootstrapped version had to be rigorously tested to ensure it correctly compiled the next version without introducing errors.
- **Unix Dependency**: C’s development was tied to Unix, which provided the environment (e.g., assembler, linker) for bootstrapping.

#### 6. **Outcome**
- By the mid-1970s, C was a mature, self-hosting language used for most of Unix, including its kernel, utilities, and tools.
- The bootstrapping process made C portable, leading to its adoption on various platforms (e.g., VAX, x86).
- The C compiler’s source code became a model for other compilers, influencing languages like C++ (built on C) and modern compilers like GCC and Clang.

---

### Example: Simplified Bootstrapping Process
To illustrate bootstrapping conceptually, consider a toy language “Mini-C” and its compiler development:

1. **Step 1: Initial Compiler in Assembly**:
   - Write a basic compiler (`compiler0`) in PDP-11 assembly that compiles a subset of Mini-C (e.g., `int`, functions, `printf`).
   - Source: `compiler0.s` (assembly).
   - Output: `compiler0` (binary).

2. **Step 2: Mini-C Compiler in Mini-C**:
   - Write a Mini-C compiler (`compiler1.c`) in the Mini-C subset.
   - Use `compiler0` to compile `compiler1.c`:
     ```bash
     compiler0 compiler1.c -o compiler1
     ```
   - Output: `compiler1` (binary, understands Mini-C subset).

3. **Step 3: Improve and Recompile**:
   - Enhance `compiler1.c` to create `compiler2.c`, adding features (e.g., `struct`, pointers).
   - Use `compiler1` to compile `compiler2.c`:
     ```bash
     compiler1 compiler2.c -o compiler2
     ```
   - Output: `compiler2` (more complete).

4. **Step 4: Full Compiler**:
   - Repeat until `compilerN.c` supports the full Mini-C language.
   - `compilerN` compiles itself:
     ```bash
     compilerN compilerN.c -o compilerN+1
     ```
   - If `compilerN+1` is identical to `compilerN`, the compiler is stable and self-hosting.

**C Example (Simplified)**:
```c
// compiler1.c (Mini-C subset)
#include <stdio.h>
int main(int argc, char* argv[]) {
    printf("Mini-C Compiler v1\n");
    // Basic parsing and codegen logic
    return 0;
}
```
- **Initial Compile**: Use an assembly/B compiler to compile `compiler1.c`.
- **Next Iteration**: Write `compiler2.c` in Mini-C, compile with `compiler1`, and so on.

This mirrors how Ritchie bootstrapped C, starting with a B/assembly compiler and iteratively building a self-hosting C compiler.

---

### Bootstrapping in Modern C Compilers
Modern C compilers like **GCC** and **Clang** continue to use bootstrapping:
- **GCC**:
  - Written in **C** and **C++**.
  - Bootstrapped in stages:
    1. Use an existing C/C++ compiler (e.g., an older GCC or Clang) to compile GCC’s source code.
    2. The resulting GCC binary compiles itself (stage 1).
    3. The stage 1 GCC compiles itself again (stage 2).
    4. If stage 2 matches stage 1, the compiler is verified as correct.
  - Repository: `https://gcc.gnu.org/git.html` (mirrors on GitHub, e.g., `https://github.com/gcc-mirror/gcc`).
- **Clang**:
  - Written in **C++**.
  - Bootstrapped similarly, often using an existing Clang or GCC to compile Clang’s source.
  - Repository: `https://github.com/llvm/llvm-project`.
- **musl libc**:
  - The C standard library is written in **C**, bootstrapped by compiling with an existing C compiler.
  - Repository: `https://github.com/musl-libc/musl`.

**Portability Example**:
To port GCC to a new architecture (e.g., RISC-V):
1. Cross-compile GCC’s source on an existing platform (e.g., x86) to produce a RISC-V binary.
2. Use the RISC-V GCC to compile itself on the target platform.
3. The resulting compiler is native and self-hosting.

---

### Relevance to Your Questions
- **C Architecture**: Bootstrapping is integral to C’s architecture, as it made C self-sustaining and portable, key to its minimalist and efficient design.
- **Language C Is Written In**: The initial C compiler was in **assembly** and **B**, but modern compilers are in **C/C++**, as bootstrapping enabled C to compile itself.
- **GitHub URLs**:
  - Clang: `https://github.com/llvm/llvm-project` (C/C++ compiler).
  - musl: `https://github.com/musl-libc/musl` (C standard library).
  - GCC Mirror: `https://github.com/gcc-mirror/gcc` (unofficial, official at `https://gcc.gnu.org/git.html`).
- **C++ Context**: Bootstrapping C laid the foundation for C++ (developed by Bjarne Stroustrup as “C with Classes”). C++ compilers (e.g., G++) are also bootstrapped, often using C as a base language.
- **Integration with `std::vector`, `{fmt}`**:
  - A C compiler doesn’t use C++ features like `std::vector` or `{fmt}`, but a modern C++ compiler (e.g., Clang) might use `std::vector` in its implementation for managing symbol tables or AST nodes.
  - Example (hypothetical Clang snippet):
    ```cpp
    #include <vector>
    #include <fmt/core.h>
    struct Node { std::string name; };
    std::vector<Node*> nodes;
    void print_node(Node* n) { fmt::print("Node: {}\n", n->name); }
    ```
    - Bootstrapping Clang involves compiling such C++ code with an existing compiler.

---

### Historical Context and Sources
- **Dennis Ritchie’s Account**: In “The Development of the C Language” (1993), Ritchie describes bootstrapping C from B and assembly, emphasizing Unix’s role.
- **Unix History**: The Unix kernel’s transition from assembly to C (1973–1975) drove C’s bootstrapping, as documented in Bell Labs archives.
- **K&R Book**: *The C Programming Language* (1978) formalized C, reflecting the bootstrapped compiler’s capabilities.
- **Search Results**: The provided results (e.g., Wikipedia, GitHub) confirm C’s origins in B/BCPL and the use of bootstrapping in compilers like GCC and Clang. No single “C repository” exists, but Clang (`https://github.com/llvm/llvm-project`) and musl (`https://github.com/musl-libc/musl`) are relevant.

---

### Best Practices for Bootstrapping
1. **Start Simple**: Begin with a minimal language subset to reduce initial compiler complexity.
2. **Test Rigorously**: Verify each bootstrapped compiler to prevent bugs from propagating.
3. **Document Stages**: Clearly document each compiler version and its supported features.
4. **Leverage Existing Tools**: Use an existing compiler (e.g., GCC, Clang) to bootstrap the initial version.
5. **Plan for Portability**: Design the compiler to support cross-compilation for new architectures.

---

### Conclusion
**Bootstrapping** is the process of building a compiler in the language it compiles, making it self-hosting and portable. For the **C language**, bootstrapping began with an initial compiler written in **assembly** and **B** (1971–1972), which compiled a C subset. This compiler was used to build increasingly complete C compilers written in C, achieving full self-hosting by 1973–1975. This process enabled C to power Unix and become a cornerstone of system programming. Modern C compilers (e.g., GCC, Clang) continue this tradition, written in **C/C++** and hosted on repositories like `https://github.com/llvm/llvm-project` (Clang) and `https://github.com/musl-libc/musl` (musl). Bootstrapping ties directly to C’s architecture, emphasizing its portability and efficiency.

If you need a specific example (e.g., a toy C compiler), help with bootstrapping a project, or more details on C’s history or modern compilers, let me know!