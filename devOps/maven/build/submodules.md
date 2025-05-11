To build only **one submodule** of a multi-module Maven project (with a `parent pom.xml`), you can use the `-pl` (projects list) and `-am` (also make dependencies) flags.

### Syntax:

```bash
mvn clean install -pl submodule-name -am
```

### Explanation:

* `-pl` (or `--projects`) specifies which module(s) to build.
* `-am` (or `--also-make`) builds any required dependencies of that module.
* You must run the command from the directory where the `parent pom.xml` resides.

### Example:

Assuming you have this structure:

```
parent-project/
├── pom.xml
├── module-a/
│   └── pom.xml
├── module-b/
│   └── pom.xml
```

To build only `module-b`:

```bash
mvn clean install -pl module-b -am
```

This will build `module-b` and its required dependencies (`module-a` if `b` depends on it), but **not the whole project**.

Would you like to skip tests or run in a specific profile too?
