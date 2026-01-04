# Rules

* **Rules** are explicit constraints that control how the model should behave and respond.
* They define **what the model can and cannot do** (e.g., don’t guess, don’t use external knowledge).
* Rules improve **consistency and reliability** of outputs.
* They help reduce **hallucinations and off-topic responses**.
* In prompt engineering, rules act like **guardrails** around the model’s reasoning.


### Best Practices 

* Write rules **clearly and unambiguously**, using simple language.
* Place rules in a **system or instruction prompt**, not in user content.
* Keep rules **short, specific, and enforceable**.
* Use **“must / must not”** wording instead of vague guidance.
* Separate **rules, context, and task** into distinct sections.
* Prefer **constraints over suggestions**.
* Include a rule for **handling missing or uncertain information**.
* Avoid conflicting or overlapping rules.
* Validate rules with **edge-case examples**.
* Review and update rules as business requirements evolve.


**Interview question:**
*How would you design and enforce prompt rules to ensure a model never uses external knowledge beyond the provided context, and how would you validate that the rule is consistently followed?*



I would place a **strict grounding rule** in the system prompt stating that the model must use only the provided context and must explicitly say “I don’t know” if information is missing.
I’d reinforce this with **output constraints** (structured formats, citations required) so violations are detectable.
To validate, I’d run **negative tests** with missing or misleading inputs and check for hallucinations.
I’d also log and monitor responses in production to catch drift.
