Covering **all design patterns** (Creational, Structural, Behavioral) is a big task, but here’s a solid overview with **Java code examples** for the most common ones:

---

## 🏗️ Creational Patterns

### 1. **Singleton**

Ensure a class has only one instance.

```java
public class Singleton {
    private static final Singleton INSTANCE = new Singleton();
    private Singleton() {}
    public static Singleton getInstance() { return INSTANCE; }
}
```

### 2. **Factory Method**

Delegate instantiation to subclasses.

```java
interface Animal { void speak(); }
class Dog implements Animal { public void speak() { System.out.println("Bark"); } }
class Cat implements Animal { public void speak() { System.out.println("Meow"); } }

abstract class AnimalFactory {
    abstract Animal create();
}

class DogFactory extends AnimalFactory {
    Animal create() { return new Dog(); }
}

class CatFactory extends AnimalFactory {
    Animal create() { return new Cat(); }
}
```

### 3. **Abstract Factory**

Create families of related objects.

```java
interface GUIFactory {
    Button createButton();
    Checkbox createCheckbox();
}
class WinFactory implements GUIFactory {
    Button createButton() { return new WinButton(); }
    Checkbox createCheckbox() { return new WinCheckbox(); }
}
class MacFactory implements GUIFactory {
    Button createButton() { return new MacButton(); }
    Checkbox createCheckbox() { return new MacCheckbox(); }
}
```

### 4. **Builder**

Construct complex objects step by step.

```java
class User {
    private String name; private int age;
    private User(UserBuilder b) { name = b.name; age = b.age; }
    static class UserBuilder {
        private String name; private int age;
        UserBuilder name(String n){ name = n; return this;}
        UserBuilder age(int a){ age = a; return this;}
        User build(){ return new User(this); }
    }
}
new User.UserBuilder().name("Alice").age(30).build();
```

### 5. **Prototype**

Clone objects.

```java
class Sheep implements Cloneable {
    String name;
    public Sheep clone() throws CloneNotSupportedException {
        return (Sheep) super.clone();
    }
}
```

---

## 🧱 Structural Patterns

### 6. **Adapter**

Convert one interface to another.

```java
interface Turkey { void gobble(); void fly(); }
class WildTurkey implements Turkey { ... }

interface Duck { void quack(); void fly(); }
class TurkeyAdapter implements Duck {
    Turkey turkey;
    TurkeyAdapter(Turkey t) { turkey = t; }
    public void quack() { turkey.gobble(); }
    public void fly() { turkey.fly(); }
}
```

### 7. **Decorator**

Add behavior dynamically.

```java
interface Coffee { double cost(); }
class SimpleCoffee implements Coffee { public double cost() { return 5; } }
abstract class CoffeeDecorator implements Coffee {
    protected Coffee coffee;
    CoffeeDecorator(Coffee c) { coffee = c; }
}
class MilkDecorator extends CoffeeDecorator {
    MilkDecorator(Coffee c){ super(c); }
    public double cost(){ return coffee.cost() + 2; }
}
```

### 8. **Facade**

Simplify a complex subsystem.

```java
class CPU { void freeze(){} void jump(long pos){} void execute(){} }
class Memory { void load(long pos, byte[] data){} }
class HardDrive { byte[] read(long lba, int size){ return new byte[size];} }
class ComputerFacade {
    private CPU cpu = new CPU();
    private Memory memory = new Memory();
    private HardDrive hd = new HardDrive();
    public void start() {
        cpu.freeze();
        memory.load(...);
        cpu.jump(...);
        cpu.execute();
    }
}
```

### 9. **Proxy**

Control access to an object.

```java
interface Image { void display(); }
class RealImage implements Image {
    private String fileName;
    RealImage(String fn){ fileName = fn; load(); }
    public void display() { System.out.println("Display " + fileName); }
    private void load() { System.out.println("Loading " + fileName); }
}
class ProxyImage implements Image {
    private RealImage realImage;
    private String fileName;
    ProxyImage(String fn){ fileName = fn; }
    public void display(){
        if(realImage == null) realImage = new RealImage(fileName);
        realImage.display();
    }
}
```

---

## 🧠 Behavioral Patterns

### 10. **Observer**

Publish–subscribe model.

```java
interface Observer { void update(String msg); }
interface Subject {
    void register(Observer o);
    void unregister(Observer o);
    void notifyObservers(String msg);
}
class NewsAgency implements Subject {
    List<Observer> observers = new ArrayList<>();
    public void register(Observer o) { observers.add(o); }
    public void unregister(Observer o) { observers.remove(o); }
    public void notifyObservers(String msg) {
       observers.forEach(o -> o.update(msg));
    }
}
```

### 11. **Strategy**

Choose algorithm at runtime.

```java
interface Strategy { int doOperation(int a, int b); }
class Add implements Strategy { public int doOperation(int a, int b){ return a+b;} }
class Context {
    Strategy strategy;
    Context(Strategy s){ strategy = s; }
    int execute(int a, int b){ return strategy.doOperation(a,b); }
}
```

### 12. **Command**

Encapsulate requests as objects.

```java
interface Command { void execute(); }
class Light {
    public void on() { System.out.println("Light on"); }
    public void off() { System.out.println("Light off"); }
}
class LightOnCommand implements Command {
    Light light;
    LightOnCommand(Light l){ light = l; }
    public void execute(){ light.on(); }
}
class RemoteControl {
    private Command slot;
    void setCommand(Command c){ slot = c; }
    void buttonPressed(){ slot.execute(); }
}
```

### 13. **Chain of Responsibility**

Pass request along a chain.

```java
abstract class Handler {
    protected Handler next;
    public void setNext(Handler h){ next = h; }
    public abstract void handle(int req);
}
class DivisibleBy2 extends Handler {
    public void handle(int req) {
        if(req % 2 == 0) System.out.println("Handled by 2");
        else if(next != null) next.handle(req);
    }
}
```

### 14. **Iterator**

Traverse without exposing structure.

```java
class NameRepository {
    private String[] names = {"Alice", "Bob", "Charlie"};
    public Iterator<String> getIterator(){ return new NameIterator(); }
    private class NameIterator implements Iterator<String> {
        int index = 0;
        public boolean hasNext(){ return index < names.length; }
        public String next(){ return names[index++]; }
    }
}
```

### 15. **Template Method**

Define skeleton, let subclasses fill in.

```java
abstract class Game {
    abstract void initialize();
    abstract void startPlay();
    abstract void endPlay();
    public final void play(){
        initialize();
        startPlay();
        endPlay();
    }
}
class Cricket extends Game {
    void initialize(){ System.out.println("Cricket init"); }
    void startPlay(){ System.out.println("Cricket start"); }
    void endPlay(){ System.out.println("Cricket end"); }
}
```

### 16. **State**

Change behavior based on state.

```java
interface State { void handle(); }
class OnState implements State { public void handle(){ System.out.println("Device ON"); } }
class OffState implements State { public void handle(){ System.out.println("Device OFF"); } }
class ContextWithState {
    private State state;
    ContextWithState(State s){ state = s; }
    void setState(State s){ state = s; }
    void request(){ state.handle(); }
}
```

### 17. **Mediator**

Centralizes communication.

```java
interface Mediator { void notify(Component sender, String event); }
class ConcreteMediator implements Mediator {
    ComponentA a; ComponentB b;
    public void notify(Component sender, String event){
        if(sender == a) b.doSomething();
    }
}
abstract class Component {
    Mediator mediator;
    Component(Mediator m){ mediator = m; }
}
class ComponentA extends Component {
    ComponentA(Mediator m){ super(m); }
    void doAction(){ mediator.notify(this, "A"); }
}
```

---

## 📊 Pattern Summary

* **Creational**: Singleton, Factory, Abstract Factory, Builder, Prototype
* **Structural**: Adapter, Decorator, Facade, Proxy, Composite (omitted due to space)
* **Behavioral**: Observer, Strategy, Command, Chain, Iterator, Template, State, Mediator

---

These are among the most widely used design patterns in Java. Let me know if you'd like a deeper dive into any pattern, or a working GitHub repo showcasing them all!
