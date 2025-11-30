

### **JVM-internals-only table** (no language-level or library-level features).


| Java Version      | JVM Internal Features Added                          | JVM Enhancements / Tuning                          | JVM Features Deprecated / Removed                         |
| ----------------- | ---------------------------------------------------- | -------------------------------------------------- | --------------------------------------------------------- |
| **Java 8**        | Metaspace replaces PermGen                           | ParallelGC, CMS tuning                             | PermGen removed                                           |
| **Java 9**        | Modular JVM (Jigsaw), Compact Strings                | G1 becomes default GC                              | No major removals                                         |
| **Java 10**       | Unified GC Interface                                 | Application CDS (AOT-friendly)                     | No major removals                                         |
| **Java 11**       | Flight Recorder + JFR Event Streaming                | ZGC experimental release                           | Java EE/CORBA removal indirectly simplifies JVM modules   |
| **Java 12**       | Abortable Mixed Collections (G1), C1/C2 improvements | JVMTI improvements                                 | No major removals                                         |
| **Java 13**       | Dynamic CDS Archives                                 | Rewritten NIO Socket layer                         | Pack200 deprecated (affects class file toolchain)         |
| **Java 14**       | Enhanced NPE for diagnostics                         | G1 region pinning improvements                     | Solaris/SPARC Port removed                                |
| **Java 15**       | Hidden Classes for dynamic frameworks                | ZGC & Shenandoah improvements                      | Nashorn removed; RMI Activation deprecated                |
| **Java 16**       | Elastic Metaspace                                    | Vector API (incubator); CDS archiving              | Deprecated `Thread.stop()` removed                        |
| **Java 17 (LTS)** | Strong encapsulation of JDK internals                | New macOS Metal rendering pipeline                 | Applet API deprecated (JVM plugin stack effectively dead) |
| **Java 18**       | UTF-8 default across JVM runtime                     | JDK Flight Recorder tuning                         | Security Manager marked for removal                       |
| **Java 19**       | Virtual Threads runtime support (Loom)               | Enhanced Foreign Function & Memory internals       | No major removals                                         |
| **Java 20**       | Scoped Values (runtime primitive for Loom)           | Virtual Thread scheduler refinement                | Deprecation clean-up of old APIs                          |
| **Java 21 (LTS)** | Finalized Virtual Threads, new Thread Scheduler      | ZGC improvements; Region Pinning Removal           | Security Manager removed                                  |
| **Java 22**       | Structured Concurrency runtime                       | Vector API iteration, JIT tuning                   | Finalization scheduled for removal                        |
| **Java 23**       | Class-File API (replace ASM dependency patterns)     | G1/ZGC/OpenJ9 improvements; VThread runtime tuning | Finalization disabled by default                          |

### **Java language–syntax–only features table** 
 

| Java Version      | New Language Syntax Introduced                                       | Syntax Enhancements                             | Syntax Deprecated / Removed                                              |
| ----------------- | -------------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------ |
| **Java 8**        | Lambda expressions `()->{}`, Method references `::`, Default methods | Functional interfaces inferred automatically    | No syntax removed                                                        |
| **Java 9**        | Module declarations (`module {}`)                                    | Private interface methods                       | No syntax removed                                                        |
| **Java 10**       | `var` for local variable type inference                              | Improved type inference for anonymous classes   | No syntax removed                                                        |
| **Java 11**       | `var` allowed in lambda parameters                                   | New string literal helpers (not syntax)         | No syntax removed                                                        |
| **Java 12**       | Switch Expressions (preview) → `switch(x) -> ...`                    | Arrow-based switch rules                        | No syntax removed                                                        |
| **Java 13**       | Text blocks `""" ... """` (preview)                                  | Escape sequences improved                       | No syntax removed                                                        |
| **Java 14**       | Records (preview) `record Point(int x, int y)`                       | Pattern matching for `instanceof` (preview)     | No syntax removed                                                        |
| **Java 15**       | Sealed classes (preview) `permits`                                   | Text blocks become standard                     | No syntax removed                                                        |
| **Java 16**       | Records (standard)                                                   | Pattern Matching (standard)                     | No syntax removed                                                        |
| **Java 17 (LTS)** | Sealed classes (standard)                                            | Switch pattern matching preview                 | Applet syntax irrelevant → effectively dead                              |
| **Java 18**       | No new syntax                                                        | UTF-8 default — not syntax                      | No syntax removed                                                        |
| **Java 19**       | Record Patterns (preview), `case Point(int x, int y)`                | Switch with pattern matching updated            | No syntax removed                                                        |
| **Java 20**       | Record Patterns (2nd preview)                                        | Pattern matching refinements                    | No syntax removed                                                        |
| **Java 21 (LTS)** | Unnamed Classes & Instance Main (preview)                            | Pattern Matching (standard), Switch refinements | Security Manager related syntax deprecated earlier (not language syntax) |
| **Java 22**       | No new syntax                                                        | Stream-like pattern updates (not syntax)        | No syntax removed                                                        |
| **Java 23**       | No new major syntax                                                  | Class-File API (not language syntax)            | Finalization removed → affects old finalize() usage but not syntax       |


 



More:

* A **Mermaid timeline of JVM evolution**
* A **separate table only for GC evolution across versions**
* A **separate table only for JIT/C2/C1 improvements**
 
* A separate table **only for pattern matching–related features**
* A **syntax-only cheat sheet** with examples


