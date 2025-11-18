Random Number Generation in Java

How Java Generates Random Numbers Internally

Java’s random generation is handled by the JVM using two main mechanisms:

1. java.util.Random

Uses a Linear Congruential Generator (LCG).

State is a 48-bit seed stored inside the JVM heap.

Each call updates seed as:
seed = (seed * 25214903917 + 11) & ((1L << 48) - 1)

The JVM executes this computation in pure Java code without native system calls.


2. java.security.SecureRandom

Uses OS entropy via /dev/urandom on Linux or NativePRNG.

In Linux, the entropy is sourced from the Linux Kernel Random API.

Entropy is mixed using hashing algorithms inside the JCE provider.


Java Example (LCG Based)

Random random = new Random();
int value = random.nextInt(100);  // 0 to 99

Java Pseudocode (LCG Internals)

seed = initial_seed
function next(bits):
    seed = (seed * 25214903917 + 11) mod 2^48
    return seed >> (48 - bits)

Real Use Case

Pagination token generation in a Java microservice that does not require cryptographic security uses java.util.Random directly in JVM memory.

Random Number Generation in Python

How Python Generates Random Numbers Internally

Python’s random module is powered by:

1. Mersenne Twister Algorithm (MT19937)

State is 624 integers maintained in CPython runtime.

All operations are done in C inside CPython’s _randommodule.c.

Not suitable for cryptographic use.


2. secrets module

Cryptographically secure RNG.

Uses OS-level CSPRNG, typically Linux Kernel Random API (/dev/urandom).

CPython calls POSIX syscalls through getrandom().


Python Example (Mersenne Twister)

import random
value = random.randint(0, 99)

Python Pseudocode (MT Internals)

state[0..623]
function twist():
    for i in 0..623:
        temp = combine_bits(state[i], state[(i+1) % 624])
        state[i] = state[(i + 397) % 624] ^ (temp >> 1) ^ (temp & 1 ? MATRIX_A : 0)

function extract():
    if exhausted: twist()
    y = tempering(state[index])
    index += 1
    return y

Real Use Case

A Python script that distributes load across Kafka partitions can use random.randint() for non-critical pseudo-random partition selection.

Summary of Components Performing the Action

Java

JVM executes LCG inside java.util.Random.

JCE + Linux Kernel Random API supply entropy for SecureRandom.


Python

CPython C runtime implements Mersenne Twister.

Linux Kernel Random API generates entropy for secrets and OS CSPRNG.


Kubernetes Randomness Usage

Kubernetes uses randomness mainly for scheduling backoff and retry logic.
The kubelet, kube-controller-manager, and kube-apiserver rely on Golang’s math/rand (LCG-like PRNG) for non-critical randomness.
Examples include pod crash-loop backoff, leader election jitter, and node-status update jitter.
Pseudo code:

delay = base + randFloat()*jitter
sleep(delay)

Real use case: kubelet adds jitter to pod restart backoff to avoid synchronized restarts across nodes.

Linux Randomness Usage

Linux provides randomness via the Kernel Random API exposing /dev/random, /dev/urandom, and getrandom() system call.
Entropy sources include interrupts, timings, scheduler jitter, and device events.
The kernel’s ChaCha20-based DRNG expands this entropy into pseudorandom data.
Example:

int fd = open("/dev/urandom", O_RDONLY);
read(fd, buf, 32);

Real use case: TLS handshakes in system services fetch randomness using getrandom().

Oracle Database Randomness Usage

Oracle uses randomness in the Cost-Based Optimizer (CBO) during table sampling and statistics estimation.
DBMS_RANDOM uses an internal PRNG implemented inside the Oracle RDBMS kernel, seeded per-session.
Example:

SELECT DBMS_RANDOM.VALUE(0, 100) FROM dual;

CBO use case: dynamic sampling picks random blocks to estimate cardinality and join selectivity.

Kafka Randomness Usage

Kafka uses randomness in partition assignment and producer ID generation.
The Kafka Producer client uses Java’s ThreadLocalRandom (fast LCG variant inside JVM) to pick a random partition when no key is provided.
Example:

partition = random.nextInt(numPartitions)

Use case: events without keys get load-distributed across partitions based on random selection.

Java Randomness Usage

Java applies randomness via the JVM using LCG (java.util.Random) and SecureRandom backed by the Linux Kernel Random API.
Example:

Random r = new Random();
int x = r.nextInt(100);

Use case: microservices generate non-secure IDs or perform randomized algorithms such as reservoir sampling.

Python Randomness Usage

Python’s random uses Mersenne Twister implemented in CPython C runtime, while secrets and os.urandom() rely on Linux Kernel Random API.
Example:

import random
random.randint(1, 10)

Use case: load testing scripts generating random payloads or selecting random Kafka partitions.