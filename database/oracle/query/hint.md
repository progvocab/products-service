Here’s a detailed explanation of **hints** in Oracle and a non-exhaustive but comprehensive list of commonly used hints. If you tell me your Oracle version (e.g. 11g, 12c, 19c etc.), I can give you the hints particularly relevant to that version.

---

## What are Hints in Oracle

* A *hint* is a directive you give inside an SQL statement to influence the Oracle optimizer’s choice of execution plan. ([Oracle Documentation][1])

* Hints can help when the optimizer’s statistics, estimates, or default behavior isn’t producing a good plan. But hints are double-edged: they can help now but may hurt later when data distributions, indexes etc. change. ([Redgate Software][2])

* Syntax: you embed hints in a comment right after the DML/SELECT etc keyword, with the `+` style, e.g.:

  ```sql
  SELECT /*+ FIRST_ROWS(10) USE_NL(emp dept) */ e.name, d.name
  FROM emp e, dept d
  WHERE e.dept_id = d.dept_id;
  ```

* Oracle supports many categories of hints: Access paths (indexes vs full scans), join methods, join order, parallelism, query transformations (e.g. unnesting subqueries), optimization goals, etc. ([Oracle Documentation][1])

---

## Categories of Hints & Key Examples

Here are the major categories (i.e. what kind of aspect of query optimization they affect) with common hints in each, with a brief description. Not every hint works in every version; availability & behavior can vary.

| Category                                     | Hint / Examples                                                                                                                                                                                                                                                                                                                                                                        | Purpose / What it does                                                                                                                                                                           |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Optimization goal / approach**             | `ALL_ROWS`; `FIRST_ROWS(n)`; `RULE` (old/rule-based); `CHOOSE` (historical) ([Oracle Documentation][1])                                                                                                                                                                                                                                                                                | Whether to optimize for throughput (fetch all rows with minimal total cost) vs response time / first rows fast. RULE forces rule-based optimizer, mostly deprecated. ([Oracle Documentation][1]) |
| **Access path hints**                        | `FULL(table)`; `INDEX(table index_name)`; `NO_INDEX(table index_name)`; `INDEX_ASC`, `INDEX_DESC`; `HASH(table)`, `CLUSTER(table)`; `INDEX_JOIN`, `INDEX_COMBINE`; skip scans etc. ([Oracle Documentation][3])                                                                                                                                                                         | Force or forbid certain ways for Oracle to access data for a table (full table scan vs index vs combinations etc.).                                                                              |
| **Join order / join order hints**            | `LEADING(tab1 tab2 …)`; `ORDERED` ([Oracle Documentation][3])                                                                                                                                                                                                                                                                                                                          | Force a particular join order: which table(s) should be joined first etc. `ORDERED` uses the order in the FROM clause. `LEADING` more explicit.                                                  |
| **Join method hints**                        | `USE_NL(table)` / `NO_USE_NL(table)` (Nested Loops); `USE_HASH(table)` / `NO_USE_HASH(table)` (Hash Join); `USE_MERGE(table)` / `NO_USE_MERGE(table)` (Merge Join); `USE_NL_WITH_INDEX` ([Oracle Documentation][3])                                                                                                                                                                    | Forces or forbids particular join algorithm for joining tables. Useful when optimizer picks a less desirable method.                                                                             |
| **Parallel execution**                       | `PARALLEL(table, degree)`; `NO_PARALLEL(table)`; `PARALLEL_INDEX` / `NO_PARALLEAL_INDEX`; `PQ_DISTRIBUTE` etc. ([Oracle Documentation][1])                                                                                                                                                                                                                                             | Enable/disable or suggest degree of parallelism (how many parallel execution slaves) etc.                                                                                                        |
| **Query block / subquery / transformations** | `UNNEST` / `NO_UNNEST`; `MERGE` / `NO_MERGE`; `STAR_TRANSFORMATION` / `NO_STAR_TRANSFORMATION`; `USE_CONCAT`; `NO_EXPAND`; `REWRITE` / `NO_REWRITE`; `FACT` / `NO_FACT`; `NO_QUERY_TRANSFORMATION` ([Oracle Documentation][4])                                                                                                                                                         | These affect how subqueries are flattened, views merged, how query transformations (e.g. pushing predicates, rewriting via materialized views etc.) are applied.                                 |
| **Miscellaneous / tuning / other hints**     | `CACHE(table)` / `NOCACHE(table)` (buffer cache behavior); `APPEND` / `NOAPPEND` (direct path insert vs conventional insert); `CARDINALITY(table, value)` (override row count estimate for a table); `DYNAMIC_SAMPLING` (level of dynamic sampling of data/statistics); `QB_NAME` (give a name to query block, useful when using hints in nested subqueries or views) ([psoug.org][5]) |                                                                                                                                                                                                  |

---

## Why Hints Sometimes “Don’t Work” or Don’t Reduce Cost

* A hint is **just a suggestion**; if the hint is impossible (index doesn’t exist, etc.) Oracle ignores it. ([Oracle Documentation][1])
* The optimizer might still override hints under certain conditions (version, compatibility, for example when certain optimizer features are disabled or overridden by system settings).
* If the hint forces a plan that isn't optimal with current data stats, it may increase cost.
* Hints require maintenance: as schemas, data, stats change, a hint that was good earlier may become suboptimal.

---

## List of Commonly Used Hints (≈ partial list)

Here’s a more or less full list of the hints you’ll see often / are documented. Not all are in every version.

```
ALL_ROWS
FIRST_ROWS(n)
RULE
CHOOSE

FULL(table)
INDEX(table [index_name…])
NO_INDEX(table [index_name…])
INDEX_ASC
INDEX_DESC
INDEX_JOIN
INDEX_COMBINE
CLUSTER(table)
HASH(table)

LEADING(tbl1 tbl2 …)
ORDERED

USE_NL(table [other tables])
NO_USE_NL(table [other tables])
USE_HASH(table [other tables])
NO_USE_HASH(table [other tables])
USE_MERGE(table [other tables])
NO_USE_MERGE(table [other tables])
USE_NL_WITH_INDEX(table index_name)

UNNEST / NO_UNNEST
MERGE / NO_MERGE
STAR_TRANSFORMATION / NO_STAR_TRANSFORMATION
USE_CONCAT
FACT / NO_FACT
NO_QUERY_TRANSFORMATION
REWRITE / NO_REWRITE
NO_EXPAND

PARALLEL(table, degree)
NO_PARALLEL(table)
PARALLEL_INDEX
NO_PARALLEL_INDEX
PQ_DISTRIBUTE

CACHE(table)
NOCACHE(table)
APPEND / NOAPPEND
CARDINALITY(table, n)
DYNAMIC_SAMPLING
QB_NAME(query_block_name)

DRIVING_SITE
CURSOR_SHARING_EXACT
MONITOR / NO_MONITOR
```

---

If you like, I can pull up the **full hint reference for your version** (say Oracle 19c) so you have a definitive list. Do you want me to do that?

[1]: https://docs.oracle.com/cd/E15586_01/server.1111/e16638/hintsref.htm?utm_source=chatgpt.com "19 Using Optimizer Hints"
[2]: https://www.red-gate.com/simple-talk/databases/oracle-databases/a-beginners-guide-to-optimizer-hints/?utm_source=chatgpt.com "A Beginner's Guide to Optimizer Hints - Simple Talk"
[3]: https://docs.oracle.com/cd/B12037_01/server.101/b10752/hintsref.htm?utm_source=chatgpt.com "17 Optimizer Hints"
[4]: https://docs.oracle.com/cd/B19306_01/server.102/b14211/hintsref.htm?utm_source=chatgpt.com "16 Using Optimizer Hints"
[5]: https://psoug.org/reference/hints.html?utm_source=chatgpt.com "Oracle Hints SQL PL/SQL"
