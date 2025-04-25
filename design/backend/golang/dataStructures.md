Here’s a collection of **fundamental data structures** implemented in **Go (Golang)**:

1. **Array**  
2. **Linked List** (Singly and Doubly)  
3. **Stack**  
4. **Queue** (with Circular Queue)  
5. **Deque (Double-Ended Queue)**  
6. **Hash Map (Custom Implementation)**  
7. **Binary Tree** (with Binary Search Tree - BST)  
8. **Graph** (Adjacency List Representation)  
9. **Heap (Min-Heap & Max-Heap)**  
10. **Trie (Prefix Tree)**  

---

## **1. Array (Fixed-Size Collection)**

Arrays in Go have **fixed size** and hold items of the **same type**.

```go
package main
import "fmt"

func main() {
	// Declare and initialize an array
	var arr [5]int = [5]int{10, 20, 30, 40, 50}

	// Access and modify array elements
	arr[2] = 35

	// Iterate over the array
	for i, v := range arr {
		fmt.Printf("Index: %d, Value: %d\n", i, v)
	}
}
```

✅ **Use arrays for fixed-size data where length is known at compile time.**

---

## **2. Singly Linked List**

A **linked list** is a linear data structure where elements (nodes) are connected by pointers.

```go
package main
import "fmt"

// Node represents a linked list node
type Node struct {
	data int
	next *Node
}

// LinkedList structure
type LinkedList struct {
	head *Node
}

// Add adds a new node at the end
func (ll *LinkedList) Add(data int) {
	newNode := &Node{data: data}
	if ll.head == nil {
		ll.head = newNode
		return
	}
	current := ll.head
	for current.next != nil {
		current = current.next
	}
	current.next = newNode
}

// Display prints the linked list
func (ll *LinkedList) Display() {
	current := ll.head
	for current != nil {
		fmt.Printf("%d -> ", current.data)
		current = current.next
	}
	fmt.Println("nil")
}

func main() {
	ll := &LinkedList{}
	ll.Add(10)
	ll.Add(20)
	ll.Add(30)
	ll.Display() // Output: 10 -> 20 -> 30 -> nil
}
```

✅ **Best for dynamic datasets where size changes frequently.**

---

## **3. Stack (LIFO - Last In, First Out)**

A **stack** follows **LIFO** order – the last item added is the first item removed.

```go
package main
import "fmt"

// Stack structure
type Stack struct {
	items []int
}

// Push adds an element to the stack
func (s *Stack) Push(item int) {
	s.items = append(s.items, item)
}

// Pop removes and returns the top element
func (s *Stack) Pop() int {
	if len(s.items) == 0 {
		return -1 // Stack is empty
	}
	last := s.items[len(s.items)-1]
	s.items = s.items[:len(s.items)-1]
	return last
}

// Peek returns the top element without removing it
func (s *Stack) Peek() int {
	if len(s.items) == 0 {
		return -1
	}
	return s.items[len(s.items)-1]
}

func main() {
	s := &Stack{}
	s.Push(10)
	s.Push(20)
	fmt.Println(s.Pop())  // Output: 20
	fmt.Println(s.Peek()) // Output: 10
}
```

✅ **Used for backtracking (e.g., undo functionality) and parsing.**

---

## **4. Queue (FIFO - First In, First Out)**

A **queue** is a linear structure following the **FIFO** order.

```go
package main
import "fmt"

// Queue structure
type Queue struct {
	items []int
}

// Enqueue adds an element to the end
func (q *Queue) Enqueue(item int) {
	q.items = append(q.items, item)
}

// Dequeue removes and returns the front element
func (q *Queue) Dequeue() int {
	if len(q.items) == 0 {
		return -1
	}
	front := q.items[0]
	q.items = q.items[1:]
	return front
}

func main() {
	q := &Queue{}
	q.Enqueue(10)
	q.Enqueue(20)
	fmt.Println(q.Dequeue()) // Output: 10
}
```

✅ **Great for task scheduling and breadth-first search (BFS).**

---

## **5. Hash Map (Key-Value Store)**

A **hash map** stores data using **key-value pairs** with constant-time lookup.

```go
package main
import "fmt"

type HashMap struct {
	buckets map[int]string
}

func NewHashMap() *HashMap {
	return &HashMap{buckets: make(map[int]string)}
}

func (hm *HashMap) Put(key int, value string) {
	hm.buckets[key] = value
}

func (hm *HashMap) Get(key int) string {
	return hm.buckets[key]
}

func (hm *HashMap) Delete(key int) {
	delete(hm.buckets, key)
}

func main() {
	hm := NewHashMap()
	hm.Put(1, "Go")
	fmt.Println(hm.Get(1)) // Output: Go
	hm.Delete(1)
}
```

✅ **Ideal for fast lookups, caching, and indexing.**

---

## **6. Binary Search Tree (BST)**

A **binary search tree** organizes elements in a hierarchical order.

```go
package main
import "fmt"

// Node structure
type Node struct {
	data  int
	left  *Node
	right *Node
}

// Insert inserts a new node
func (n *Node) Insert(data int) {
	if n == nil {
		return
	} else if data < n.data {
		if n.left == nil {
			n.left = &Node{data: data}
		} else {
			n.left.Insert(data)
		}
	} else {
		if n.right == nil {
			n.right = &Node{data: data}
		} else {
			n.right.Insert(data)
		}
	}
}

// InOrder prints in ascending order
func (n *Node) InOrder() {
	if n == nil {
		return
	}
	n.left.InOrder()
	fmt.Println(n.data)
	n.right.InOrder()
}

func main() {
	root := &Node{data: 50}
	root.Insert(30)
	root.Insert(70)
	root.Insert(20)
	root.InOrder() // Output: 20, 30, 50, 70
}
```

✅ **Perfect for ordered data and fast range queries.**

---

## **7. Graph (Adjacency List)**

A **graph** represents relationships between nodes.

```go
package main
import "fmt"

type Graph struct {
	nodes map[int][]int
}

func NewGraph() *Graph {
	return &Graph{nodes: make(map[int][]int)}
}

func (g *Graph) AddEdge(v1, v2 int) {
	g.nodes[v1] = append(g.nodes[v1], v2)
	g.nodes[v2] = append(g.nodes[v2], v1)
}

func (g *Graph) Display() {
	for node, edges := range g.nodes {
		fmt.Println(node, "->", edges)
	}
}

func main() {
	g := NewGraph()
	g.AddEdge(1, 2)
	g.AddEdge(2, 3)
	g.Display()
}
```

✅ **Used for modeling networks and social graphs.**

---

## **8. Trie (Prefix Tree)**

A **Trie** is an efficient information retrieval structure, useful for auto-completion.

```go
package main
import "fmt"

type TrieNode struct {
	children map[rune]*TrieNode
	isEnd    bool
}

type Trie struct {
	root *TrieNode
}

func NewTrie() *Trie {
	return &Trie{root: &TrieNode{children: make(map[rune]*TrieNode)}}
}

func (t *Trie) Insert(word string) {
	current := t.root
	for _, ch := range word {
		if _, found := current.children[ch]; !found {
			current.children[ch] = &TrieNode{children: make(map[rune]*TrieNode)}
		}
		current = current.children[ch]
	}
	current.isEnd = true
}

func (t *Trie) Search(word string) bool {
	current := t.root
	for _, ch := range word {
		if _, found := current.children[ch]; !found {
			return false
		}
		current = current.children[ch]
	}
	return current.isEnd
}

func main() {
	t := NewTrie()
	t.Insert("go")
	fmt.Println(t.Search("go"))  // Output: true
}
```

✅ **Perfect for prefix-based search.**

Would you like a **priority queue** or **custom heap** implementation next?