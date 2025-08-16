Database normalization is a process used in relational database design to minimize data redundancy and improve data integrity. The goal is to break down complex tables into simpler ones and establish relationships between them using foreign keys.

Normalization involves organizing the columns (attributes) and tables (relations) of a database to ensure that dependencies are properly enforced by database integrity constraints.

---

## 🔄 **Types of Normal Forms (NF)**

Here are the types of normal forms with definitions and SQL examples.

---

### ✅ **1NF (First Normal Form)**
**Rule**: Eliminate repeating groups; each cell should contain only atomic (indivisible) values.

**Example (Unnormalized Table)**

| StudentID | Name   | Subjects             |
|-----------|--------|----------------------|
| 1         | Alice  | Math, Science        |
| 2         | Bob    | English, History     |

**Convert to 1NF**

| StudentID | Name   | Subject   |
|-----------|--------|-----------|
| 1         | Alice  | Math      |
| 1         | Alice  | Science   |
| 2         | Bob    | English   |
| 2         | Bob    | History   |

```sql
CREATE TABLE Students1NF (
    StudentID INT,
    Name VARCHAR(50),
    Subject VARCHAR(50)
);
```

---

### ✅ **2NF (Second Normal Form)**
**Rule**: Be in 1NF and have no partial dependency (no non-prime attribute should depend on a part of a composite key).

**Example (1NF with partial dependency)**


| StudentID | CourseID | StudentName | CourseName |
|-----------|----------|-------------|------------|
| 1         | 1  | Alics      | Math|


Here, `StudentName` depends only on `StudentID`, and `CourseName` only on `CourseID`.

**Split into 2NF**

- **Students Table**  
```sql
CREATE TABLE Students (
    StudentID INT PRIMARY KEY,
    StudentName VARCHAR(50)
);
```

- **Courses Table**  
```sql
CREATE TABLE Courses (
    CourseID INT PRIMARY KEY,
    CourseName VARCHAR(50)
);
```

- **Enrollment Table**  
```sql
CREATE TABLE Enrollment (
    StudentID INT,
    CourseID INT,
    PRIMARY KEY (StudentID, CourseID),
    FOREIGN KEY (StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY (CourseID) REFERENCES Courses(CourseID)
);
```

---

### ✅ **3NF (Third Normal Form)**
**Rule**: Be in 2NF and have no transitive dependencies (non-prime attributes should not depend on other non-prime attributes).

**Example (2NF with transitive dependency)**

| EmpID | EmpName | DeptID | DeptName |
|-------|---------|--------|----------|
| 1          | Alics |1     | IT|


Here, `DeptName` depends on `DeptID`, which is not a primary key.

**Split into 3NF**

- **Employees Table**
```sql
CREATE TABLE Employees (
    EmpID INT PRIMARY KEY,
    EmpName VARCHAR(50),
    DeptID INT,
    FOREIGN KEY (DeptID) REFERENCES Departments(DeptID)
);
```

- **Departments Table**
```sql
CREATE TABLE Departments (
    DeptID INT PRIMARY KEY,
    DeptName VARCHAR(50)
);
```

---

### ✅ **BCNF (Boyce-Codd Normal Form)**
**Rule**: A stricter version of 3NF. Every determinant must be a candidate key.

**Example (3NF but not BCNF)**

| StudentID | Course | Instructor |
|-----------|--------|------------|
| 1 | Math | Ray |

Assume:  
- Each course has only one instructor.  
- StudentID and Course is the PK, but Course → Instructor

**Problem**: Instructor depends on Course, not on the full PK.

**Solution**: Split into:

- **Courses Table**
```sql
CREATE TABLE Courses (
    Course VARCHAR(50) PRIMARY KEY,
    Instructor VARCHAR(50)
);
```

- **Enrollment Table**
```sql
CREATE TABLE Enrollment (
    StudentID INT,
    Course VARCHAR(50),
    PRIMARY KEY (StudentID, Course),
    FOREIGN KEY (Course) REFERENCES Courses(Course)
);
```

---

### ✅ **4NF (Fourth Normal Form)**
**Rule**: Be in BCNF and have no multi-valued dependencies.

**Example**

| Student | Hobby      | Language   |
|---------|------------|------------|
| Alice   | Painting   | English    |
| Alice   | Painting   | Spanish    |
| Alice   | Swimming   | English    |
| Alice   | Swimming   | Spanish    |

Here, Hobby and Language are independent but paired with Student, causing a multi-valued dependency.

**Split into:**

```sql
CREATE TABLE StudentHobbies (
    Student VARCHAR(50),
    Hobby VARCHAR(50),
    PRIMARY KEY (Student, Hobby)
);

CREATE TABLE StudentLanguages (
    Student VARCHAR(50),
    Language VARCHAR(50),
    PRIMARY KEY (Student, Language)
);
```

---

### ✅ **5NF (Fifth Normal Form / Project-Join Normal Form)**
**Rule**: Be in 4NF and cannot be further decomposed without loss of data.

Used in very rare, complex scenarios involving join dependencies.

---

### ✅ **6NF (Sixth Normal Form)**
Used in temporal databases. Every join dependency is a trivial join dependency. Rarely used in practice.

---

Would you like to see this with a real-world example, like an e-commerce or HR database?
