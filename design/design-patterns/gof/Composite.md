# **Composite Design Pattern in Golang (With Real-World Scenarios)**  

## **What is the Composite Pattern?**  
The **Composite Pattern** is a structural design pattern used to treat individual objects and groups of objects **uniformly**. It creates a tree structure where both **leaf nodes** (individual objects) and **composite nodes** (groups of objects) implement the same interface.  

### **Real-World Scenarios**  
1. **File System (Files & Folders)**
2. **Organization Structure (Employees & Managers)**
3. **Graphics Editor (Shapes & Groups of Shapes)**  

---

## **1. Example: File System (Files & Folders)**
Imagine a file system where **folders** contain **files or other folders**.  

### **Golang Implementation:**
```go
package main

import (
	"fmt"
)

// Component Interface
type FileSystemComponent interface {
	ShowDetails(indent string)
}

// Leaf: File
type File struct {
	Name string
}

func (f *File) ShowDetails(indent string) {
	fmt.Println(indent + "📄 File: " + f.Name)
}

// Composite: Folder
type Folder struct {
	Name        string
	Components  []FileSystemComponent
}

func (f *Folder) Add(component FileSystemComponent) {
	f.Components = append(f.Components, component)
}

func (f *Folder) ShowDetails(indent string) {
	fmt.Println(indent + "📁 Folder: " + f.Name)
	for _, component := range f.Components {
		component.ShowDetails(indent + "  ")
	}
}

// Main Function
func main() {
	root := &Folder{Name: "Root"}
	
	folder1 := &Folder{Name: "Documents"}
	file1 := &File{Name: "Resume.pdf"}
	file2 := &File{Name: "Project.docx"}

	folder2 := &Folder{Name: "Pictures"}
	file3 := &File{Name: "Photo.jpg"}

	folder1.Add(file1)
	folder1.Add(file2)
	folder2.Add(file3)

	root.Add(folder1)
	root.Add(folder2)

	root.ShowDetails("")
}
```
### **Output:**
```
📁 Folder: Root
  📁 Folder: Documents
    📄 File: Resume.pdf
    📄 File: Project.docx
  📁 Folder: Pictures
    📄 File: Photo.jpg
```

---

## **2. Example: Organization Structure (Employees & Managers)**
An **organization** has employees and managers. A **manager can have employees under them**, forming a tree structure.

### **Golang Implementation:**
```go
package main

import "fmt"

// Component Interface
type Employee interface {
	ShowDetails(indent string)
}

// Leaf: Individual Employee
type Developer struct {
	Name string
}

func (d *Developer) ShowDetails(indent string) {
	fmt.Println(indent + "👨‍💻 Developer: " + d.Name)
}

// Composite: Manager with Subordinates
type Manager struct {
	Name       string
	Employees  []Employee
}

func (m *Manager) Add(employee Employee) {
	m.Employees = append(m.Employees, employee)
}

func (m *Manager) ShowDetails(indent string) {
	fmt.Println(indent + "👨‍💼 Manager: " + m.Name)
	for _, emp := range m.Employees {
		emp.ShowDetails(indent + "  ")
	}
}

// Main Function
func main() {
	// Creating Employees
	dev1 := &Developer{Name: "Alice"}
	dev2 := &Developer{Name: "Bob"}

	manager := &Manager{Name: "Charlie"}
	manager.Add(dev1)
	manager.Add(dev2)

	ceo := &Manager{Name: "Eve"}
	ceo.Add(manager)

	// Display Organization Structure
	ceo.ShowDetails("")
}
```
### **Output:**
```
👨‍💼 Manager: Eve
  👨‍💼 Manager: Charlie
    👨‍💻 Developer: Alice
    👨‍💻 Developer: Bob
```

---

## **3. Example: Graphics Editor (Shapes & Groups of Shapes)**
A **graphics editor** allows you to group shapes (like circles, rectangles) and apply transformations to **individual shapes or entire groups**.

### **Golang Implementation:**
```go
package main

import "fmt"

// Component Interface
type Shape interface {
	Draw(indent string)
}

// Leaf: Circle
type Circle struct {
	Name string
}

func (c *Circle) Draw(indent string) {
	fmt.Println(indent + "⭕ Circle: " + c.Name)
}

// Leaf: Rectangle
type Rectangle struct {
	Name string
}

func (r *Rectangle) Draw(indent string) {
	fmt.Println(indent + "▭ Rectangle: " + r.Name)
}

// Composite: Group of Shapes
type ShapeGroup struct {
	Name   string
	Shapes []Shape
}

func (g *ShapeGroup) Add(shape Shape) {
	g.Shapes = append(g.Shapes, shape)
}

func (g *ShapeGroup) Draw(indent string) {
	fmt.Println(indent + "🎨 Shape Group: " + g.Name)
	for _, shape := range g.Shapes {
		shape.Draw(indent + "  ")
	}
}

// Main Function
func main() {
	circle1 := &Circle{Name: "Red Circle"}
	circle2 := &Circle{Name: "Blue Circle"}
	rect1 := &Rectangle{Name: "Green Rectangle"}

	group1 := &ShapeGroup{Name: "Group 1"}
	group1.Add(circle1)
	group1.Add(rect1)

	mainGroup := &ShapeGroup{Name: "Main Group"}
	mainGroup.Add(group1)
	mainGroup.Add(circle2)

	mainGroup.Draw("")
}
```
### **Output:**
```
🎨 Shape Group: Main Group
  🎨 Shape Group: Group 1
    ⭕ Circle: Red Circle
    ▭ Rectangle: Green Rectangle
  ⭕ Circle: Blue Circle
```

---

## **Key Takeaways**
✅ The **Composite Pattern** lets you treat individual objects and groups **uniformly**.  
✅ It's useful for **hierarchical structures** like file systems, organizations, and graphics editors.  
✅ **Composite objects** contain other components, and **leaf objects** are the actual elements.  
✅ This pattern **simplifies client code**, as it can handle individual and group objects **in the same way**.  

Would you like an example for a **different real-world scenario**?