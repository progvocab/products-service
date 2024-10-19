The **Gang of Four (GoF) Design Patterns** refer to 23 classic software design patterns introduced in the book *"Design Patterns: Elements of Reusable Object-Oriented Software"*, written by **Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides**. These four authors are collectively known as the "Gang of Four," and their book became a foundational reference for software developers to create flexible, reusable, and maintainable object-oriented systems.

The 23 GoF patterns are divided into three main categories:

1. **Creational Patterns**  
   Deal with object creation mechanisms, trying to create objects in a manner suitable to the situation. These patterns abstract the instantiation process, making a system independent of how its objects are created, composed, and represented.

   - **Factory Method**: Define an interface for creating an object, but let subclasses decide which class to instantiate. It allows a class to defer instantiation to subclasses.
   - **Abstract Factory**: Provides an interface for creating families of related or dependent objects without specifying their concrete classes.
   - **Builder**: Separate the construction of a complex object from its representation, allowing the same construction process to create different representations.
   - **Prototype**: Specify the kinds of objects to create using a prototypical instance, and create new objects by copying this prototype.
   - **Singleton**: Ensure a class has only one instance and provide a global point of access to it.

2. **Structural Patterns**  
   Concerned with how classes and objects are composed to form larger structures. These patterns help ensure that if one part of a system changes, the entire structure doesn't need to change.

   - **Adapter**: Convert the interface of a class into another interface clients expect. The Adapter pattern allows classes to work together that couldn't otherwise because of incompatible interfaces.
   - **Bridge**: Separate an object’s abstraction from its implementation so that the two can vary independently.
   - **Composite**: Compose objects into tree structures to represent part-whole hierarchies. Composite lets clients treat individual objects and compositions of objects uniformly.
   - **Decorator**: Attach additional responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing for extending functionality.
   - **Facade**: Provide a unified interface to a set of interfaces in a subsystem. The Facade pattern defines a higher-level interface that makes the subsystem easier to use.
   - **Flyweight**: Use sharing to support a large number of fine-grained objects efficiently.
   - **Proxy**: Provide a surrogate or placeholder for another object to control access to it.

3. **Behavioral Patterns**  
   Concerned with algorithms and the assignment of responsibilities between objects. These patterns deal with how objects interact and communicate.

   - **Chain of Responsibility**: Pass a request along a chain of handlers. Upon receiving a request, each handler decides either to process the request or to pass it to the next handler in the chain.
   - **Command**: Encapsulate a request as an object, thereby allowing for parameterization of clients with different requests, queuing of requests, and logging of the requests.
   - **Interpreter**: Define a representation for a language's grammar, along with an interpreter that uses the representation to interpret sentences in the language.
   - **Iterator**: Provide a way to access the elements of an aggregate object sequentially without exposing its underlying representation.
   - **Mediator**: Define an object that encapsulates how a set of objects interact. Mediator promotes loose coupling by keeping objects from referring to each other explicitly.
   - **Memento**: Without violating encapsulation, capture and externalize an object’s internal state so that the object can be restored to this state later.
   - **Observer**: Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.
   - **State**: Allow an object to alter its behavior when its internal state changes. The object will appear to change its class.
   - **Strategy**: Define a family of algorithms, encapsulate each one, and make them interchangeable. The strategy lets the algorithm vary independently from clients that use it.
   - **Template Method**: Define the skeleton of an algorithm in an operation, deferring some steps to subclasses. Template Method lets subclasses redefine certain steps of an algorithm without changing the algorithm’s structure.
   - **Visitor**: Represent an operation to be performed on the elements of an object structure. Visitor lets you define a new operation without changing the classes of the elements on which it operates.

### Purpose of Design Patterns
Design patterns are not complete designs that can be directly converted into code. Instead, they provide solutions to common design problems, offering a blueprint to write flexible, reusable, and maintainable software. The key benefits include:
- **Reusability**: Patterns allow you to reuse solutions that have been proven to work.
- **Maintainability**: Patterns can make the system more maintainable by decoupling components, thus reducing dependencies.
- **Flexibility**: They allow systems to be more flexible in the face of changing requirements.

These patterns are highly useful in the design of scalable and robust software systems, providing a shared language and efficient solutions to recurring design problems.
