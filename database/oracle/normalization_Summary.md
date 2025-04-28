Awesome, let’s continue carefully — you have added **new columns**:

- Manager Name
- Office Location
- Employee Skill
- Employee Projects (multiple projects)
- Certification (multiple certifications)

Now, let’s **go from UNF → 6NF again** based on the **new columns**.

---

# **0. UNNORMALIZED FORM (UNF)**  
New sample raw data:

| Employee Name | Salary | City | Zip | Reportees | Designation | Grade | Manager Name | Office Location | Skill | Projects         | Certifications   |
|:--------------|:-------|:-----|:----|:----------|:------------|:------|:-------------|:----------------|:------|:-----------------|:-----------------|
| Bob           | 1000   | NY   |10001| Alex\|Jenny| Manager     | A     | John         | Building A      | Java  | P1\|P2           | AWS\|Azure       |

---
# **1. FIRST NORMAL FORM (1NF)**  
**Problem:** Multi-valued fields (Projects, Certifications, Reportees).

**Fix:** Flatten into atomic values:  
Each Project, Certification, Reportee should have separate rows.

| Employee Name | Salary | City | Zip  | Reportee | Designation | Grade | Manager Name | Office Location | Skill | Project | Certification |
|:--------------|:-------|:-----|:-----|:---------|:------------|:------|:-------------|:----------------|:------|:--------|:--------------|
| Bob           | 1000   | NY   |10001 | Alex     | Manager     | A     | John         | Building A      | Java  | P1      | AWS           |
| Bob           | 1000   | NY   |10001 | Alex     | Manager     | A     | John         | Building A      | Java  | P1      | Azure         |
| Bob           | 1000   | NY   |10001 | Jenny    | Manager     | A     | John         | Building A      | Java  | P2      | AWS           |
| Bob           | 1000   | NY   |10001 | Jenny    | Manager     | A     | John         | Building A      | Java  | P2      | Azure         |

(Massive duplication happens.)

---
# **2. SECOND NORMAL FORM (2NF)**  
**Problem:** Partial dependency (e.g., Project Role dependent on Project, Skill dependent on Employee).

**Fix:** Decompose:

- **Employee Table** → Personal details.
- **Project Table** → Project info.
- **Certification Table** → Certification info.
- **Skill Table** → Skills separately.

### New Tables:

**Employee Table**

| Employee Name | Salary | Zip  | Manager Name | Office Location | Grade |
|:--------------|:-------|:-----|:-------------|:----------------|:------|
| Bob           | 1000   |10001 | John         | Building A      | A     |

**Employee-Reportee Table**

| Employee Name | Reportee |
|:--------------|:---------|
| Bob           | Alex     |
| Bob           | Jenny    |

**Employee-Project Table**

| Employee Name | Project ID |
|:--------------|:-----------|
| Bob           | P1          |
| Bob           | P2          |

**Employee-Certification Table**

| Employee Name | Certification |
|:--------------|:--------------|
| Bob           | AWS           |
| Bob           | Azure         |

**Employee-Skill Table**

| Employee Name | Skill |
|:--------------|:------|
| Bob           | Java  |

---
# **3. THIRD NORMAL FORM (3NF)**  
**Problem:** Transitive dependencies.

- **City depends on Zip Code** (Zip → City).
- **Designation depends on Grade** (Grade → Designation).
- **Office Location depends on Manager**? (Assuming not — else we fix it).

**Fix:** Create lookup tables:

**Zip Code Table**

| Zip  | City |
|:-----|:-----|
|10001 | NY   |

**Grade Table**

| Grade | Designation |
|:------|:------------|
| A     | Manager     |

---
# **4. BOYCE-CODD NORMAL FORM (BCNF)**  
**Fix:**  
If Manager → Office Location (one-to-one mapping), then separate.

**Manager Table**

| Manager Name | Office Location |
|:-------------|:----------------|
| John         | Building A       |

---
# **5. FOURTH NORMAL FORM (4NF)**  
**Problem:** Multivalued facts still exist independently:  
- Reportees
- Projects
- Certifications

**Fix:** Keep separate mapping tables (already done in 2NF above).

No additional change needed.

---
# **6. FIFTH NORMAL FORM (5NF)**  
**Problem:** If Project Roles or Certification bodies become independent, need to further split.

Example:  
- Same employee can have different roles in different projects.
- Different authorities issue certifications.

Further normalize if needed:
- Employee → Project → Role
- Employee → Certification → Issuer

**(optional depending on business rules)**

---
# **7. SIXTH NORMAL FORM (6NF)**  
**Problem:**  
Time-varying attributes (Skills, Salary, Grade promotions, Certifications expiry).

**Fix:**  
Add validity columns:

**Employee-Salary History Table**

| Employee Name | Salary | Valid_From | Valid_To |
|:--------------|:-------|:------------|:---------|
| Bob           | 1000   |2024-01-01   |2024-06-01|
| Bob           | 1200   |2024-06-01   |Present   |

**Employee-Grade History Table**

| Employee Name | Grade | Valid_From | Valid_To |
|:--------------|:------|:------------|:---------|
| Bob           | A     |2024-01-01   |Present   |

**Employee-Skill History Table**

| Employee Name | Skill | Valid_From | Valid_To |
|:--------------|:------|:------------|:---------|
| Bob           | Java  |2024-01-01   |Present   |

---

# **Final list of Tables (Fully Normalized to 6NF)**

| Table | Description |
|:------|:------------|
| Employee | Base attributes |
| Employee-Reportee | Employee to Reportee |
| Employee-Project | Employee to Project |
| Employee-Certification | Employee to Certifications |
| Employee-Skill | Employee Skills |
| Salary History | Employee Salary over time |
| Grade History | Grade changes over time |
| Manager | Manager's office location |
| Zip Code | Zip to City mapping |
| Project | Project information |
| Certification | Certification details |

---

# **Summary of Key New Concepts**

| Step | New Columns Effect |
|:-----|:-------------------|
| 1NF | Atomicity (split Reportees, Projects, Certifications) |
| 2NF | Removed partial dependencies (split Skills, Projects) |
| 3NF | Removed transitive dependencies (Zip → City, Grade → Designation) |
| BCNF | Manager → Office Location if needed |
| 4NF | Multi-valued facts into separate tables |
| 5NF | Independence of Skills, Projects, Certifications |
| 6NF | Time-varying data (Salary, Grade, Skills history) |

---

Would you also like me to now show a **diagram (ERD)** based on this 6NF structure?  
It will be even easier to visualize the entire setup!  

**Shall I proceed to show ERD for you?** (It would look awesome and professional!)