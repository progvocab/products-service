The `@Transactional` annotation in **Java** is part of the **Spring Framework** and is used to manage transactions declaratively. It simplifies transaction management by allowing developers to define transactional boundaries around methods or classes without needing to write boilerplate code for beginning and committing transactions explicitly.

### **Key Features of `@Transactional`**:

1. **Declarative Transaction Management**: Automatically handles transaction boundaries, making the code cleaner and more manageable.
2. **Rollbacks**: Automatically rolls back the transaction in case of runtime exceptions.
3. **Propagation Behavior**: Defines how transactions should propagate across method calls.
4. **Isolation Level**: Controls the visibility of transaction changes to other transactions.
5. **Timeouts**: Specifies how long a transaction can run before it should be rolled back.
6. **Read-Only**: Optimizes read-only transactions by avoiding unnecessary locks.

### **Basic Usage**:

1. **Annotating a Method**:
   - You can annotate a method with `@Transactional` to define the transactional behavior for that specific method.

   ```java
   @Service
   public class UserService {

       @Transactional
       public void createUser(User user) {
           // code to save user to the database
       }
   }
   ```

2. **Annotating a Class**:
   - You can also annotate a class, which applies the transactional behavior to all methods within the class.

   ```java
   @Service
   @Transactional
   public class UserService {
       
       public void createUser(User user) {
           // code to save user to the database
       }
   }
   ```

### **Attributes of `@Transactional`**:

1. **`propagation`**: Defines how transactions relate to each other.
   - **`REQUIRED`** (default): Uses the current transaction or creates a new one if none exists.
   - **`REQUIRES_NEW`**: Suspends the current transaction and creates a new one.
   - **`MANDATORY`**: Requires an existing transaction; throws an exception if none exists.
   - **`SUPPORTS`**: Runs within a transaction if one exists, otherwise runs non-transactionally.
   - **`NOT_SUPPORTED`**: Runs non-transactionally, suspending any existing transaction.
   - **`NEVER`**: Throws an exception if a transaction exists.
   - **`NESTED`**: Executes within a nested transaction if a transaction exists.

2. **`isolation`**: Sets the isolation level for the transaction.
   - **`DEFAULT`**: Uses the default isolation level of the database.
   - **`READ_UNCOMMITTED`**: Allows dirty reads, non-repeatable reads, and phantom reads.
   - **`READ_COMMITTED`**: Prevents dirty reads, allows non-repeatable reads and phantom reads.
   - **`REPEATABLE_READ`**: Prevents dirty and non-repeatable reads, allows phantom reads.
   - **`SERIALIZABLE`**: Prevents dirty reads, non-repeatable reads, and phantom reads.

3. **`timeout`**: Specifies the timeout for the transaction in seconds.

   ```java
   @Transactional(timeout = 30)
   public void processData() {
       // transactional code
   }
   ```

4. **`readOnly`**: Optimizes the transaction for read-only operations.

   ```java
   @Transactional(readOnly = true)
   public List<User> fetchUsers() {
       // code to fetch users
   }
   ```

5. **`rollbackFor` and `noRollbackFor`**: Specifies the exceptions that should trigger a rollback or not.

   ```java
   @Transactional(rollbackFor = CustomException.class)
   public void performAction() throws CustomException {
       // transactional code
   }
   ```

### **Example of Advanced Usage**:

```java
@Transactional(
    propagation = Propagation.REQUIRED,
    isolation = Isolation.READ_COMMITTED,
    timeout = 20,
    readOnly = false,
    rollbackFor = {SQLException.class}
)
public void transferFunds(Account fromAccount, Account toAccount, BigDecimal amount) throws SQLException {
    // Code to transfer funds
}
```

### **Transaction Propagation Example**:

- **Scenario**: Service A calls Service B, both methods are transactional.

```java
@Service
public class ServiceA {
    
    @Autowired
    private ServiceB serviceB;

    @Transactional
    public void methodA() {
        // Business logic for Service A
        serviceB.methodB(); // Calls methodB in Service B
    }
}

@Service
public class ServiceB {

    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void methodB() {
        // Business logic for Service B
    }
}
```

- **Explanation**: In this scenario, `methodA` will run in a new transaction or an existing one, while `methodB` will always run in a new transaction, suspending the one from `methodA`.

### **Considerations and Best Practices**:

1. **Exception Handling**: By default, `@Transactional` rolls back on unchecked exceptions (subclasses of `RuntimeException`). For checked exceptions, you need to explicitly specify rollback behavior using `rollbackFor`.
2. **Transaction Management**: Use `@Transactional` judiciously, especially in performance-critical applications, as it can introduce overhead.
3. **Database-Specific Behavior**: Be aware of the underlying database’s transaction management and isolation levels.
4. **Method Visibility**: Ensure that methods annotated with `@Transactional` are `public` as Spring AOP proxies can only apply to public methods.

### **Conclusion**:
`@Transactional` in Spring simplifies transaction management in Java applications by handling the complexities of transaction boundaries, rollbacks, and propagation. It allows developers to focus on business logic while ensuring data integrity and consistency in transactional operations.