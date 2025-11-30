

# Java

### [JVM (Java Virtual Machine)](jvm)
  The engine that actually executes Java bytecode — it loads class files, verifies them, manages memory (heap, stack), and performs Just-In-Time (JIT) compilation.
- [Interpreter and JIT Compilers](jvm/Interpreter.md)
### **JRE (Java Runtime Environment):**
  A **package** that provides everything needed to run Java programs — it includes the **JVM**, **core libraries**, and **runtime files**, but **not** development tools (like `javac`).


 
 - **JVM** = Execution engine
 - **JRE** = JVM + Libraries (to *run* Java)
 - **JDK** = JRE + Compiler & Tools (to *develop* Java)
 
### [Classloader](classloader/)
Classloader is component within JVM responsible for loading the class in the memory.
### [Memory Management](memory/)
Depending on the garbage collector chosen JVM divides the heap memory in Regions or Generations.
### [Multi Threading](multi_threading/)
Multiple Thread management strategies are available using synchronized , Locks , atomic References and Virtul threads.

### [Versions](versions/)
 

| Java Version      | New Features Added                                             | Updates / Enhancements                                   | Deprecated / Removed Features                             |
| ----------------- | -------------------------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------- |
| [**Java 8**](versions/java8)        | Lambdas, Streams API, Optional, Default Methods, Date/Time API | StampedLock, Concurrent Updates, Metaspace               | Permanent Generation removed                              |
| **Java 9**        | JPMS (Modules), JShell, Multi-release JARs                     | Streams & Optional enhancements, Compact Strings         | Applet API deprecated; JavaFX removed from JDK later      |
| **Java 10**       | `var` local variable inference                                 | Improved GC Interface; Application Class Data Sharing    | No major removals                                         |
| [**Java 11**](versions/java11)       | Standard HTTP Client                                           | Flight Recorder, String API improvements                 | Java EE & CORBA removed; Nashorn deprecated               |
| **Java 12**       | Switch Expressions (preview)                                   | JVMTI updates, Collectors.teeing                         | No major removals                                         |
| **Java 13**       | Text Blocks (preview)                                          | Socket API rewrite                                       | Pack200 Tools/APIs deprecated                             |
| **Java 14**       | Records (preview), Pattern Matching (preview)                  | NPE detailed messages                                    | Solaris/Sparc port removed                                |
| **Java 15**       | Sealed Classes (preview), Hidden Classes                       | Edwards Curve Algorithms                                 | Nashorn removed; RMI Activation deprecated                |
| **Java 16**       | Records (standard), Packaging Tool                             | Improved Metaspace, Vector API (incubator)               | Deprecated Thread.stop removed                            |
| [**Java 17 (LTS)**](versions/java17) | Sealed Classes (final)                                         | Strong encapsulation of JDK internals                    | Applet API deprecated for removal; RMI Activation removed |
| **Java 18**       | Simple Web Server                                              | UTF-8 default charset                                    | Final removal of Security Manager initiated (deprecated)  |
| **Java 19**       | Virtual Threads (preview), Structured Concurrency              | Foreign Memory & Function API updates                    | No major removals                                         |
| **Java 20**       | Scoped Values (incubator)                                      | Pattern Matching & Virtual Threads update                | Final removal of deprecated methods/apis begun            |
| [**Java 21 (LTS)**](versions/java21) | Sequenced Collections; Unnamed Classes                         | Virtual Threads finalized; Pattern Matching improvements | Security Manager removed; Final removal of RMI Activation |
| **Java 22**       | Stream Gatherers                                               | Vector API improvements                                  | Deprecated finalization—scheduled for removal             |
| **Java 23**       | Class File API                                                 | G1/ZGC tuning; Virtual Thread enhancements               | Finalization disabled by default                          |

 
