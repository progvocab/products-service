Sure! Here's a quick guide to the **basic operations** you can perform on arrays in Java:

---

## **1. Declaration**

```java
int[] numbers;
String[] names;
```

---

## **2. Initialization**

```java
int[] numbers = new int[5];                    // default values (0)
String[] names = {"Alice", "Bob", "Charlie"};  // initialized with values
```

---

## **3. Access Elements**

```java
int first = numbers[0];        // get element at index 0
numbers[1] = 42;               // set value at index 1
```

---

## **4. Get Length**

```java
int len = numbers.length;     // no parentheses! It's a field, not a method
```

---

## **5. Iteration**

### a. Traditional for loop:
```java
for (int i = 0; i < numbers.length; i++) {
    System.out.println(numbers[i]);
}
```

### b. Enhanced for-each loop:
```java
for (int num : numbers) {
    System.out.println(num);
}
```

---

## **6. Sorting**

```java
Arrays.sort(numbers);    // Sorts in ascending order
```

---

## **7. Searching**

```java
int index = Arrays.binarySearch(numbers, 42);  // array must be sorted
```

---

## **8. Copying Arrays**

```java
int[] copy = Arrays.copyOf(numbers, numbers.length);
```

---

## **9. Filling with a Value**

```java
Arrays.fill(numbers, 7);  // sets all elements to 7
```

---

## **10. Multi-Dimensional Arrays**

```java
int[][] matrix = new int[3][3];
matrix[0][1] = 5;
```

---

## **Bonus: Converting to String**

```java
System.out.println(Arrays.toString(numbers));  // [7, 7, 7, 7, 7]
```

---

Let me know if you want examples of advanced operations like filtering or mapping with **Streams**!