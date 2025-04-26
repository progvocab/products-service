Apache Airflow provides various **subcommands** for managing DAGs using the `airflow dags` CLI command. Below is an explanation of the key subcommands:  

---

## **1. backfill**  
**Usage:**  
```bash
airflow dags backfill <dag_id> -s <start_date> -e <end_date>
```
- Runs a DAG for a specific date range (historical execution).  
- Example:  
  ```bash
  airflow dags backfill my_dag -s 2024-01-01 -e 2024-01-07
  ```
- This will run the DAG from January 1 to January 7, 2024.

---

## **2. delete**  
**Usage:**  
```bash
airflow dags delete <dag_id>
```
- Deletes a DAG from the **Airflow metadata database** (not the DAG file).  
- Example:  
  ```bash
  airflow dags delete my_dag
  ```

---

## **3. details**  
**Usage:**  
```bash
airflow dags details <dag_id>
```
- Shows detailed information about a DAG, including schedule, tasks, and dependencies.  
- Example:  
  ```bash
  airflow dags details my_dag
  ```

---

## **4. list**  
**Usage:**  
```bash
airflow dags list
```
- Lists all available DAGs in Airflow.  
- Example output:  
  ```
  DAG ID          | Schedule        | Last Run
  ------------------------------------------------
  my_dag         | @daily          | 2024-01-29 00:00:00
  another_dag    | 0 12 * * *       | 2024-01-28 12:00:00
  ```

---

## **5. list-import-errors**  
**Usage:**  
```bash
airflow dags list-import-errors
```
- Lists DAGs that failed to import due to errors in the DAG file.  
- Helps debug syntax or import issues.

---

## **6. list-jobs**  
**Usage:**  
```bash
airflow dags list-jobs
```
- Lists jobs (e.g., scheduler and backfill jobs) associated with DAGs.  
- Example:  
  ```
  ID  | DAG ID    | Job Type  | Start Date
  --------------------------------------------
  10  | my_dag    | Backfill  | 2024-01-28 12:00:00
  11  | another_dag | Scheduler | 2024-01-28 13:00:00
  ```

---

## **7. list-runs**  
**Usage:**  
```bash
airflow dags list-runs <dag_id>
```
- Lists all runs for a specific DAG, including their execution dates and statuses.  
- Example:  
  ```bash
  airflow dags list-runs my_dag
  ```
  ```
  ID  | DAG ID  | Execution Date   | State
  -----------------------------------------
  100 | my_dag  | 2024-01-27       | success
  101 | my_dag  | 2024-01-28       | failed
  ```

---

## **8. next-execution**  
**Usage:**  
```bash
airflow dags next-execution <dag_id>
```
- Shows the next scheduled execution time of a DAG.  
- Example:  
  ```bash
  airflow dags next-execution my_dag
  ```
  ```
  2024-01-30 00:00:00 UTC
  ```

---

## **9. pause**  
**Usage:**  
```bash
airflow dags pause <dag_id>
```
- Pauses a DAG (prevents it from running).  
- Example:  
  ```bash
  airflow dags pause my_dag
  ```

---

## **10. report**  
**Usage:**  
```bash
airflow dags report
```
- Generates a **summary report** of all DAGs, including active/inactive status, execution times, and errors.  
- Example output:  
  ```
  DAG ID     | Schedule  | Runs  | Last Run   | State
  --------------------------------------------------
  my_dag     | @daily    | 10    | 2024-01-28 | success
  another_dag | @weekly   | 5     | 2024-01-27 | failed
  ```

---

## **11. reserialize**  
**Usage:**  
```bash
airflow dags reserialize
```
- Forces **re-serialization** of all DAGs stored in the Airflow metadata database (used in DAG serialization mode).  
- Useful in Airflow setups using the **webserver with DAG serialization**.

---

## **12. show**  
**Usage:**  
```bash
airflow dags show <dag_id>
```
- Displays a **DAG graph view** (task dependencies in ASCII format).  
- Example:  
  ```bash
  airflow dags show my_dag
  ```
  ```
  start -> task_1 -> task_2 -> end
  ```

---

## **13. show-dependencies**  
**Usage:**  
```bash
airflow dags show-dependencies <dag_id>
```
- Shows task dependencies within a DAG in **graphical format** (prints an ASCII tree).  
- Example:  
  ```bash
  airflow dags show-dependencies my_dag
  ```
  ```
  start
    ├── task_1
    │   ├── task_2
    │   │   ├── task_3
    │   ├── task_4
  ```

---

## **14. state**  
**Usage:**  
```bash
airflow dags state <dag_id> <execution_date>
```
- Gets the **current state** of a DAG run for a specific execution date.  
- Example:  
  ```bash
  airflow dags state my_dag 2024-01-28
  ```
  ```
  success
  ```

---

## **15. test**  
**Usage:**  
```bash
airflow dags test <dag_id> <execution_date>
```
- Runs a DAG **task-by-task** without recording results in the database.  
- Useful for debugging.  
- Example:  
  ```bash
  airflow dags test my_dag 2024-01-28
  ```
  ```
  Running task_1...
  Running task_2...
  DAG completed successfully.
  ```

---

## **16. trigger**  
**Usage:**  
```bash
airflow dags trigger <dag_id>
```
- **Manually triggers** a DAG run.  
- Example:  
  ```bash
  airflow dags trigger my_dag
  ```
  ```
  DAG my_dag triggered successfully.
  ```

---

## **17. unpause**  
**Usage:**  
```bash
airflow dags unpause <dag_id>
```
- **Resumes** execution of a paused DAG.  
- Example:  
  ```bash
  airflow dags unpause my_dag
  ```

---

### **Summary Table**
| Subcommand            | Description |
|-----------------------|-------------|
| `backfill`           | Runs past DAG executions |
| `delete`             | Deletes a DAG from the database |
| `details`            | Shows details of a DAG |
| `list`               | Lists all DAGs |
| `list-import-errors` | Shows DAG import errors |
| `list-jobs`         | Lists jobs related to DAGs |
| `list-runs`         | Lists DAG runs |
| `next-execution`    | Shows the next execution time of a DAG |
| `pause`             | Pauses a DAG |
| `report`            | Generates a DAG summary |
| `reserialize`       | Reserializes DAGs (for web UI performance) |
| `show`              | Shows a DAG’s task flow in text format |
| `show-dependencies` | Shows DAG task dependencies graphically |
| `state`             | Displays the current state of a DAG run |
| `test`              | Runs a DAG for debugging (without database recording) |
| `trigger`           | Manually triggers a DAG run |
| `unpause`           | Resumes a paused DAG |

---

These commands help in managing, debugging, and monitoring DAGs in Apache Airflow. Would you like more details on any specific command?