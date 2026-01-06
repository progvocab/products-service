Top-P (Nucleus) Sampling** 

1. Top-P sampling selects tokens from a **dynamic subset** of the vocabulary.
2. The subset contains the **smallest number of tokens** whose cumulative probability ≥ **P**.
3. P is a value between **0 and 1** (commonly 0.8–0.95).
4. Tokens are first **sorted by probability** in descending order.
5. Probabilities are **accumulated until the P threshold is reached**.
6. All tokens outside this nucleus are **discarded**.
7. Sampling is performed **only within the nucleus set**.
8. This prevents sampling from **very low-probability (nonsense) tokens**.
9. The nucleus size **changes at every step** depending on model confidence.
10. When the model is confident, the nucleus is **small**.
11. When uncertain, the nucleus **expands** to allow diversity.
12. This adapts better than fixed Top-K sampling.
13. It balances **fluency and creativity**.
14. It reduces repetition and incoherent outputs.
15. It is widely used in **modern LLM inference**.

**In short:** Top-P sampling keeps generation both **controlled and flexible**.
