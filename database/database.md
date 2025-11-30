### Relation

A relation is a table in a relational database. It consists of rows (tuples) and columns (attributes). Each relation has a schema that defines attribute names and data types.

### Tuple

A tuple is a single row in a relation. It represents one record of data.

### Attribute

An attribute is a column in a relation. It defines a specific property of the tuples, with a name and data type.

### Domain

A domain is the set of valid values an attribute can take. Example: age attribute domain is integers from 0 to 120.

### Relation Schema

The schema defines the structure of a relation: relation name + attributes + data types.

### Relation Instance

The actual data present in a table at any point in time.

### Primary Key

A minimal set of attributes that uniquely identifies each tuple in a relation. It cannot contain null values.

### Candidate Key

All possible sets of attributes that can uniquely identify tuples. One of them is chosen as the primary key.

### Super Key

Any set of attributes that uniquely identifies tuples. A candidate key is a minimal super key.

### Composite Key

A key consisting of two or more attributes used together to uniquely identify a tuple.

### Foreign Key

An attribute in one relation that references the primary key of another relation, enforcing referential integrity.

### Referential Integrity

A rule ensuring that a foreign key value must either match an existing primary key or be null.

### Entity

A real-world object represented by a row in a table.

### Entity Set

A collection of similar entities; becomes a relation in a database.

### Cardinality

Number of rows in a table.

### Degree

Number of attributes (columns) in a table.

### Normalization

A process of restructuring a database to reduce redundancy and improve data integrity.

### Functional Dependency

Attribute B is functionally dependent on A if A uniquely determines B. Example: employee_id → employee_name.

### Partial Dependency

A dependency where a non-key attribute depends only on part of a composite key.

### Transitive Dependency

A dependency where attribute A determines B, and B determines C, so A indirectly determines C.

### Anomalies

Problems caused by poor schema design:

* Update anomaly
* Insert anomaly
* Delete anomaly

### Index

A data structure (typically B+ tree) used to speed up data retrieval.

### Clustered Index

Stores table rows in the same order as the index.

### Non-Clustered Index

Stores a separate structure with pointers to table rows.

### Transaction

A sequence of operations performed as a single logical unit of work.

### ACID Properties

Atomicity, Consistency, Isolation, Durability.

### Isolation Levels

Read Uncommitted, Read Committed, Repeatable Read, Serializable.

### Deadlock

Two transactions wait indefinitely for resources locked by each other.

### Locking

Mechanism to control concurrent access. Includes shared locks, exclusive locks, intent locks.

### Query Plan

Steps chosen by the database (via the CBO, Cost-Based Optimizer) to execute a SQL query efficiently.

If you want, I can add ER modeling terms, all normal forms (1NF–5NF), or examples for each term.


### Entity

A real-world object represented in a database. Example: Employee, Department.

### Entity Set

A collection of similar entities. Example: All employees form the Employee entity set.

### Weak Entity

An entity that cannot be uniquely identified without a foreign key from another (strong) entity. It depends on a strong entity.
Example: **OrderItem** depends on **Order**.

### Strong Entity

An entity that has its own primary key and does not depend on another entity.

### Attribute

A property of an entity. Example: Employee has attributes employee_id, name, salary.

### Key Attribute

An attribute that can uniquely identify an entity. Example: employee_id.

### Composite Attribute

An attribute composed of multiple sub-attributes.
Example: address → street, city, pin.

### Multivalued Attribute

An attribute that can have multiple values for a single entity.
Example: phone_numbers.

### Derived Attribute

An attribute that can be derived from other attributes.
Example: age derived from date_of_birth.

### Relationship

A logical association between two or more entity sets.
Example: Employee works in Department.

### Relationship Set

A collection of relationships of the same type.

### Degree of Relationship

Number of entity sets involved in the relationship:

* Unary (1 entity)
* Binary (2 entities)
* Ternary (3 entities)

### Cardinality of Relationship

Describes numbers of entities that can participate:

* One-to-One (1:1)
* One-to-Many (1:N)
* Many-to-One (N:1)
* Many-to-Many (M:N)

### Participation Constraint

Specifies whether the participation of an entity in a relationship is:

* **Total** (mandatory)
* **Partial** (optional)

### Weak Relationship

A relationship that involves a weak entity and depends on a strong entity.
Example: OrderItem belongs to Order.

### Identifying Relationship

A relationship in which a weak entity derives its primary key from the strong entity.
Example: (order_id + item_no) uniquely identify OrderItem.

### Non-Identifying Relationship

A relationship where the child entity has its own primary key.

### ER Diagram

A graphical representation of entities, attributes, and relationships.

### Extended ER (EER) Terms

### Specialization

Top-down approach: one entity divides into more specific sub-entities.
Example: Employee → Manager, Engineer.

### Generalization

Bottom-up approach: multiple entities combine into a general super-entity.
Example: Car, Truck → Vehicle.

### Aggregation

A relationship that treats another relationship as an abstract entity.
Used when a relationship needs to participate in another relationship.

Example:
Project assigned to a team (Team is a relationship of employees).

### Composition

Strong form of aggregation where the lifetime of the part depends on the whole.
Example: Room cannot exist without Building.

### Disjointness Constraint

Defines whether sub-entities can overlap:

* Disjoint: an entity belongs to only one subtype
* Overlapping: entity may belong to multiple subtypes

### Completeness Constraint

Specifies if all parent entities must be in a subtype:

* Total
* Partial

### Subtype Discriminator

An attribute that identifies which subtype an entity belongs to.
Example: employee_type = 'M' for Manager.

If you want, I can create:

* A complete ER diagram in Mermaid
* Examples for each term
* Relational model conversion rules (ER → tables)
