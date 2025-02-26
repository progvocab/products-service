### **Singly and Doubly Linked List in Golang**  

A **linked list** is a data structure where each node contains:  
- **Data** (value)  
- **Pointer to the next node** (and previous node in doubly linked lists)  

---

## **1. Singly Linked List (SLL) in Go**
- Each node points **only to the next node**.  
- Efficient for **insertions and deletions** but **slow for searching**.  

### **Implementation of Singly Linked List**
```go
package main

import "fmt"

type Node struct {
    data int
    next *Node
}

type LinkedList struct {
    head *Node
}

// Insert at the end
func (ll *LinkedList) append(data int) {
    newNode := &Node{data: data}
    if ll.head == nil {
        ll.head = newNode
        return
    }
    temp := ll.head
    for temp.next != nil {
        temp = temp.next
    }
    temp.next = newNode
}

// Insert at the beginning
func (ll *LinkedList) prepend(data int) {
    newNode := &Node{data: data, next: ll.head}
    ll.head = newNode
}

// Delete a node
func (ll *LinkedList) delete(data int) {
    if ll.head == nil {
        return
    }
    if ll.head.data == data {
        ll.head = ll.head.next
        return
    }
    prev := ll.head
    for prev.next != nil && prev.next.data != data {
        prev = prev.next
    }
    if prev.next != nil {
        prev.next = prev.next.next
    }
}

// Print the list
func (ll *LinkedList) printList() {
    temp := ll.head
    for temp != nil {
        fmt.Print(temp.data, " -> ")
        temp = temp.next
    }
    fmt.Println("nil")
}

func main() {
    ll := &LinkedList{}
    ll.append(10)
    ll.append(20)
    ll.prepend(5)
    ll.printList() // Output: 5 -> 10 -> 20 -> nil
    ll.delete(10)
    ll.printList() // Output: 5 -> 20 -> nil
}
```

---

## **2. Doubly Linked List (DLL) in Go**
- Each node points **to the next and previous node**.  
- More flexible but requires **extra memory for the `prev` pointer**.  

### **Implementation of Doubly Linked List**
```go
package main

import "fmt"

type DNode struct {
    data int
    prev *DNode
    next *DNode
}

type DoublyLinkedList struct {
    head *DNode
}

// Insert at the end
func (dll *DoublyLinkedList) append(data int) {
    newNode := &DNode{data: data}
    if dll.head == nil {
        dll.head = newNode
        return
    }
    temp := dll.head
    for temp.next != nil {
        temp = temp.next
    }
    temp.next = newNode
    newNode.prev = temp
}

// Insert at the beginning
func (dll *DoublyLinkedList) prepend(data int) {
    newNode := &DNode{data: data, next: dll.head}
    if dll.head != nil {
        dll.head.prev = newNode
    }
    dll.head = newNode
}

// Delete a node
func (dll *DoublyLinkedList) delete(data int) {
    if dll.head == nil {
        return
    }
    temp := dll.head
    for temp != nil && temp.data != data {
        temp = temp.next
    }
    if temp == nil {
        return
    }
    if temp.prev != nil {
        temp.prev.next = temp.next
    } else {
        dll.head = temp.next
    }
    if temp.next != nil {
        temp.next.prev = temp.prev
    }
}

// Print the list forward
func (dll *DoublyLinkedList) printForward() {
    temp := dll.head
    for temp != nil {
        fmt.Print(temp.data, " <-> ")
        temp = temp.next
    }
    fmt.Println("nil")
}

// Print the list backward
func (dll *DoublyLinkedList) printBackward() {
    temp := dll.head
    if temp == nil {
        return
    }
    for temp.next != nil {
        temp = temp.next
    }
    for temp != nil {
        fmt.Print(temp.data, " <-> ")
        temp = temp.prev
    }
    fmt.Println("nil")
}

func main() {
    dll := &DoublyLinkedList{}
    dll.append(10)
    dll.append(20)
    dll.prepend(5)
    dll.printForward()  // Output: 5 <-> 10 <-> 20 <-> nil
    dll.printBackward() // Output: 20 <-> 10 <-> 5 <-> nil
    dll.delete(10)
    dll.printForward()  // Output: 5 <-> 20 <-> nil
}
```

---

## **3. Differences Between SLL and DLL**
| Feature  | **Singly Linked List (SLL)** | **Doubly Linked List (DLL)** |
|----------|------------------|------------------|
| **Memory** | Less (1 pointer per node) | More (2 pointers per node) |
| **Traversal** | Forward only | Forward & backward |
| **Insertion/Deletion** | Needs full traversal (O(n)) | Faster (O(1) if node is known) |
| **Use Cases** | Simple data structures (e.g., stacks) | Complex operations (e.g., undo/redo) |

---

### **Key Takeaways**
✅ **Singly Linked List (SLL)** → Simple, but **one-way traversal**.  
✅ **Doubly Linked List (DLL)** → More memory, but **efficient bidirectional access**.  
✅ **Both have O(n) search time, but DLL allows faster deletions**.  

Would you like an example of a **circular linked list**?