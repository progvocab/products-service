What Is Monte Carlo Simulation

Monte Carlo simulation estimates a numerical result by performing many trials using random numbers and averaging the outcomes. It is useful when analytical solutions are hard or impossible.

How It Works Internally

A loop runs thousands or millions of iterations.
Each iteration uses a PRNG to generate random values, feeds them into a model, and records the outcome.
Results are aggregated (mean, variance, distribution) to approximate the expected value.

Components Performing the Action

In Java the JVM’s LCG (java.util.Random) or the OS-backed SecureRandom generates randomness.
In Python the CPython C runtime implements Mersenne Twister for random and Linux Kernel Random API for secrets.
In Linux the Kernel Random API provides entropy via getrandom() when required.

Common Example: Estimating PI

A quarter circle is inscribed in a unit square.
Random points (x, y) between 0 and 1 are generated.
Count how many fall inside the quarter circle (x² + y² ≤ 1).
Ratio of hits to total samples approximates PI/4.

Pseudocode (Relevant to Real Use)

count = 0
for i in 1..N:
    x = rng()    # JVM LCG or CPython MT depending on language
    y = rng()
    if x*x + y*y <= 1:
        count += 1
pi_estimate = 4 * (count / N)

Java Example

Random r = new Random();
int hits = 0;
for (int i = 0; i < 1_000_000; i++) {
    double x = r.nextDouble();
    double y = r.nextDouble();
    if (x*x + y*y <= 1) hits++;
}
double pi = 4.0 * hits / 1_000_000;

Python Example

import random
hits = 0
for _ in range(1_000_000):
    x = random.random()
    y = random.random()
    if x*x + y*y <= 1:
        hits += 1
pi = 4 * hits / 1_000_000

Real Use Case

Risk analysis in finance generates random market movements using the language’s PRNG (JVM LCG or CPython MT). Running thousands of simulations approximates portfolio loss distribution.