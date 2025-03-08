Apache Cassandra **does not use Maven (`pom.xml`)** because it is built using **Gradle**, not Maven. Instead of `pom.xml`, it has a **Gradle build system**, primarily defined in:  

- **`build.gradle.kts`** (Kotlin-based Gradle build script)  
- **`gradlew` / `gradlew.bat`** (Gradle wrapper for platform-independent builds)  

---

### **How is Cassandra Built Without `pom.xml`?**
Apache Cassandra follows a **Gradle-based build process** instead of using Maven. Here’s how the build happens:  

#### **1. Gradle Wrapper (`gradlew`)**
Cassandra includes a Gradle wrapper (`gradlew`), allowing it to be built without needing a separate Gradle installation.  

#### **2. `build.gradle.kts` (Main Build Script)**
Instead of `pom.xml`, Cassandra defines its build logic in **Gradle Kotlin DSL (`build.gradle.kts`)**, which includes:  
✅ **Dependencies** (similar to Maven’s `<dependencies>`)  
✅ **Compilation settings**  
✅ **Testing framework** (JUnit, TestNG)  
✅ **Packaging & distribution logic**  

#### **3. Steps to Build Cassandra**
To build Cassandra from source, follow these steps:

1. **Clone the repository**  
   ```sh
   git clone https://github.com/apache/cassandra.git
   cd cassandra
   ```

2. **Build using Gradle**  
   ```sh
   ./gradlew build   # For Linux/macOS
   gradlew.bat build # For Windows
   ```

3. **Run tests**  
   ```sh
   ./gradlew test
   ```

4. **Generate artifacts (JARs, distributions, etc.)**  
   ```sh
   ./gradlew assemble
   ```

---

### **Why Does Cassandra Use Gradle Instead of Maven?**
1. **Better Performance** → Gradle is faster than Maven for large projects.  
2. **Incremental Builds** → Gradle only recompiles changed files, reducing build times.  
3. **More Flexibility** → Kotlin DSL (`.kts`) allows better scripting than Maven’s XML.  
4. **Better Dependency Management** → Handles transitive dependencies efficiently.  

---

### **Comparison: Maven (`pom.xml`) vs Gradle (`build.gradle.kts`)**
| Feature         | Maven (`pom.xml`) | Gradle (`build.gradle.kts`) |
|---------------|-----------------|-----------------|
| Build System | XML-based | Kotlin/Groovy-based |
| Performance | Slower | Faster (incremental builds) |
| Dependency Management | Centralized (Maven Repos) | Flexible (Maven, Ivy, etc.) |
| Configuration | Verbose | Concise & scriptable |
| Used by Cassandra? | ❌ No | ✅ Yes |

---

### **How to Check Dependencies in Cassandra?**
Since there is no `pom.xml`, you can list dependencies using:  
```sh
./gradlew dependencies
```

Would you like help modifying Cassandra's **build.gradle.kts** for custom builds?