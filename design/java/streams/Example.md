Sure! Let's walk through how to perform these operations using **Java Streams** on a `List<Employee>`.

Assume this is our `Employee` class:

```java
public class Employee {
    private String name;
    private String department;
    private double salary;

    // constructor, getters, setters
}
```

And we have a `List<Employee>`:

```java
List<Employee> employees = List.of(
    new Employee("Alice", "IT", 70000),
    new Employee("Bob", "HR", 50000),
    new Employee("Charlie", "IT", 90000),
    new Employee("David", "Finance", 75000),
    new Employee("Eve", "HR", 60000)
);
```

---

### 1. **Employee with the Highest Salary**

```java
Optional<Employee> highestPaid = employees.stream()
    .max(Comparator.comparingDouble(Employee::getSalary));

highestPaid.ifPresent(System.out::println);
```

---

### 2. **Max Salary per Department**

```java
Map<String, Optional<Employee>> maxSalaryPerDept = employees.stream()
    .collect(Collectors.groupingBy(
        Employee::getDepartment,
        Collectors.maxBy(Comparator.comparingDouble(Employee::getSalary))
    ));

maxSalaryPerDept.forEach((dept, emp) -> 
    System.out.println(dept + " => " + emp.map(Employee::getName).orElse("None"))
);
```

If you only want the **salary values**, use:

```java
Map<String, Double> maxSalaryOnly = employees.stream()
    .collect(Collectors.groupingBy(
        Employee::getDepartment,
        Collectors.collectingAndThen(
            Collectors.maxBy(Comparator.comparingDouble(Employee::getSalary)),
            opt -> opt.map(Employee::getSalary).orElse(0.0)
        )
    ));
```

---

### 3. **Count of Employees in Each Department**

```java
Map<String, Long> countPerDept = employees.stream()
    .collect(Collectors.groupingBy(
        Employee::getDepartment,
        Collectors.counting()
    ));

countPerDept.forEach((dept, count) -> 
    System.out.println(dept + " => " + count)
);
```

---

Let me know if you also want to:
- Find **average salary per department**
- List **all employees sorted by department and salary**
- Convert to DTO or Map

Happy to add that too!