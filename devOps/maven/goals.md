In **Apache Maven**, **goals** represent specific tasks that are executed by **plugins**. Each plugin provides a set of goals that perform particular actions during the build process. Goals are tied to specific phases of the Maven lifecycle, and when you run a Maven command, you are essentially invoking one or more goals.

---

### **Understanding Maven Goals**
- **Plugins**: Maven plugins are collections of goals. For example, the `maven-compiler-plugin` provides goals like `compile` and `testCompile`.
- **Goals**: Goals are specific tasks within a plugin. For example, the `compile` goal in the `maven-compiler-plugin` compiles the main source code.
- **Phases**: Lifecycle phases (e.g., `compile`, `test`, `package`) are made up of plugin goals. When you run a phase, Maven executes all the goals bound to that phase and the preceding phases.

---

### **Common Maven Plugins and Their Goals**
Here are some of the most commonly used Maven plugins and their goals:

---

#### **1. `maven-compiler-plugin`**
   - **Purpose**: Compiles Java source code.
   - **Goals**:
     - `compile`: Compiles the main source code.
     - `testCompile`: Compiles the test source code.

   **Example**:
   ```bash
   mvn compiler:compile
   ```

---

#### **2. `maven-surefire-plugin`**
   - **Purpose**: Runs unit tests.
   - **Goals**:
     - `test`: Executes unit tests.

   **Example**:
   ```bash
   mvn surefire:test
   ```

---

#### **3. `maven-jar-plugin`**
   - **Purpose**: Packages the project into a JAR file.
   - **Goals**:
     - `jar`: Creates a JAR file from the compiled classes.

   **Example**:
   ```bash
   mvn jar:jar
   ```

---

#### **4. `maven-war-plugin`**
   - **Purpose**: Packages the project into a WAR file (for web applications).
   - **Goals**:
     - `war`: Creates a WAR file.

   **Example**:
   ```bash
   mvn war:war
   ```

---

#### **5. `maven-install-plugin`**
   - **Purpose**: Installs the built artifact into the local Maven repository.
   - **Goals**:
     - `install`: Installs the artifact.

   **Example**:
   ```bash
   mvn install:install
   ```

---

#### **6. `maven-deploy-plugin`**
   - **Purpose**: Deploys the built artifact to a remote repository.
   - **Goals**:
     - `deploy`: Deploys the artifact.

   **Example**:
   ```bash
   mvn deploy:deploy
   ```

---

#### **7. `maven-clean-plugin`**
   - **Purpose**: Cleans the project by deleting the `target` directory.
   - **Goals**:
     - `clean`: Deletes the `target` directory.

   **Example**:
   ```bash
   mvn clean:clean
   ```

---

#### **8. `maven-site-plugin`**
   - **Purpose**: Generates project documentation and reports.
   - **Goals**:
     - `site`: Generates the project site.
     - `deploy`: Deploys the generated site to a server.

   **Example**:
   ```bash
   mvn site:site
   ```

---

#### **9. `maven-dependency-plugin`**
   - **Purpose**: Manages project dependencies.
   - **Goals**:
     - `copy`: Copies dependencies to a specified directory.
     - `tree`: Displays the dependency tree.

   **Example**:
   ```bash
   mvn dependency:tree
   ```

---

#### **10. `maven-resources-plugin`**
   - **Purpose**: Handles resource files (e.g., configuration files).
   - **Goals**:
     - `resources`: Copies resources to the output directory.
     - `testResources`: Copies test resources to the test output directory.

   **Example**:
   ```bash
   mvn resources:resources
   ```

---

### **How Goals Are Bound to Phases**
Goals are bound to lifecycle phases in the `pom.xml` file. For example, the `compile` goal of the `maven-compiler-plugin` is bound to the `compile` phase by default. When you run:
```bash
mvn compile
```
Maven executes the `compile` goal of the `maven-compiler-plugin` during the `compile` phase.

---

### **Customizing Goal Bindings**
You can customize which goals are executed during specific phases in your `pom.xml`. For example:
```xml
<build>
  <plugins>
    <plugin>
      <groupId>org.apache.maven.plugins</groupId>
      <artifactId>maven-compiler-plugin</artifactId>
      <version>3.8.1</version>
      <executions>
        <execution>
          <phase>compile</phase>
          <goals>
            <goal>compile</goal>
          </goals>
        </execution>
      </executions>
    </plugin>
  </plugins>
</build>
```

---

### **Running Goals Directly**
You can run a specific goal directly by specifying the plugin and goal in the command:
```bash
mvn <plugin>:<goal>
```
For example:
```bash
mvn compiler:compile
```

---

### **Summary of Common Goals**
| Plugin                     | Goal               | Description                                      |
|----------------------------|--------------------|--------------------------------------------------|
| `maven-compiler-plugin`    | `compile`          | Compiles main source code.                       |
| `maven-compiler-plugin`    | `testCompile`      | Compiles test source code.                       |
| `maven-surefire-plugin`    | `test`             | Runs unit tests.                                 |
| `maven-jar-plugin`         | `jar`              | Packages the project into a JAR file.            |
| `maven-war-plugin`         | `war`              | Packages the project into a WAR file.            |
| `maven-install-plugin`     | `install`          | Installs the artifact into the local repository. |
| `maven-deploy-plugin`      | `deploy`           | Deploys the artifact to a remote repository.     |
| `maven-clean-plugin`       | `clean`            | Deletes the `target` directory.                  |
| `maven-site-plugin`        | `site`             | Generates project documentation.                 |
| `maven-dependency-plugin`  | `tree`             | Displays the dependency tree.                    |
| `maven-resources-plugin`   | `resources`        | Copies resources to the output directory.        |

---

By understanding and using Maven goals, you can customize and control the build process effectively. Let me know if you need further clarification!