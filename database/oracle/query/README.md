
View count of employess in each department  
```sql
select department_id , count(employee_id) from hr.employees group by department_id;
```

View count of employess in each department having count > 10 
```sql
select department_id , count(employee_id) 
from hr.employees 
group by department_id
having  count(employee_id) > 10;
```

Show Department name also
```sql
select e.department_id , d.department_name , count(employee_id) 
from hr.employees e
inner join hr.departments d on d.department_id = e.department_id 
group by e.department_id , d.department_name
having  count(employee_id) > 10;

```

RANK employees based on salary 
```sql

select employee_id , first_name ,SALARY,
RANK() over( ORDER BY SALARY DESC ) AS SAL_RANK
FROM hr.employees
order by sal_rank 
Fetch first 10 rows only;
```

|EMPLOYEE_ID|FIRST_NAME|SALARY|SAL_RANK|
|--|--|--|--|
|100|Steven|24000|1|
|101|Neena|17000|2|
|102|Lex|17000|2|
|145|John|14000|4|
|146|Karen|13500|5|
|201|Michael|13000|6|
|108|Nancy|12008|7|
|205|Shelley|12008|7|
|147|Alberto|12000|9|
|168|Lisa|11500|10|


None of the rank should be skipped 
```sql

select employee_id , first_name ,SALARY,
DENSE_RANK() over( ORDER BY SALARY DESC ) AS SAL_RANK
FROM hr.employees
order by sal_rank 
Fetch first 10 rows only;
```


|EMPLOYEE_ID|FIRST_NAME|SALARY|SAL_RANK|
|--|--|--|--|
|100|Steven|24000|1|
|101|Neena|17000|2|
|102|Lex|17000|2|
|145|John|14000|3|
|146|Karen|13500|4|
|201|Michael|13000|5|
|108|Nancy|12008|6|
|205|Shelley|12008|6|
|147|Alberto|12000|7|
|168|Lisa|11500|8|

Show only the employee with second highest salary 
```sql
SELECT * FROM (
select employee_id , first_name ,SALARY,
RANK() over( ORDER BY SALARY DESC ) AS SAL_RANK
FROM hr.employees
order by sal_rank 
Fetch first 10 rows only
) e where e.sal_rank =2
```

|EMPLOYEE_ID|FIRST_NAME|SALARY|SAL_RANK|
|--|--|--|--|
|101|Neena|17000|2|
|102|Lex|17000|2|

Department wise highest Salary 

```sql
select e.department_id , d.department_name , max(salary) 
from hr.employees e
inner join hr.departments d on d.department_id = e.department_id 
group by e.department_id , d.department_name
order by 3 desc
Fetch first 5 rows only
```


|DEPARTMENT_ID|DEPARTMENT_NAME|MAX SALARY|
|--|--|--|
|90|Executive|24000|
|80|Sales|14000|
|20|Marketing|13000|
|100|Finance|12008|
|110|Accounting|12008|

Rank by Salary , Department wise 
```sql 
select e.DEpartment_id , e.EMPLOYEE_ID ,  e.FIRST_NAME ,e.SALARY ,
RANK() over( partition by department_id  order by salary desc) as SAL_RANK
from hr.employees e
order by sal_rank
```

|DEPARTMENT_ID|EMPLOYEE_ID|FIRST_NAME|SALARY|SAL_RANK|
|--|--|--|--|--|
|10|200|Jennifer|4400|1|
|20|201|Michael|13000|1|
|20|202|Pat|6000|2|
|30|114|Den|11000|1|
|30|115|Alexander|3100|2|
|30|116|Shelli|2900|3|
|30|117|Sigal|2800|4|
|30|118|Guy|2600|5|
|30|119|Karen|2500|6|
|40|203|Susan|6500|1|
|50|121|Adam|8200|1|
|50|120|Matthew|8000|2|
|50|122|Payam|7900|3|
|50|123|Shanta|6500|4|
|50|124|Kevin|5800|5|
|50|184|Nandita|4200|6|

Find top 3 paid employees in each department 

```sql
select * from 
(select e.DEpartment_id , e.EMPLOYEE_ID ,  e.FIRST_NAME ,e.SALARY ,
RANK() over( partition by department_id  order by salary desc) as SAL_RANK
from hr.employees e
) e 
where sal_rank <= 3
order by department_id 
```

|DEPARTMENT_ID|EMPLOYEE_ID|FIRST_NAME|SALARY|SAL_RANK|
|--|--|--|--|--|
|10|200|Jennifer|4400|1|
|20|201|Michael|13000|1|
|20|202|Pat|6000|2|
|30|114|Den|11000|1|
|30|115|Alexander|3100|2|
|30|116|Shelli|2900|3|
|40|203|Susan|6500|1|
|50|121|Adam|8200|1|
|50|120|Matthew|8000|2|
|50|122|Payam|7900|3|
|60|103|Alexander|9000|1|
|60|104|Bruce|6000|2|
|60|105|David|4800|3|
|60|106|Valli|4800|3|
|70|204|Hermann|10000|1|
|80|145|John|14000|1|
|80|146|Karen|13500|2|
|80|147|Alberto|12000|3|
|90|100|Steven|24000|1|
|90|101|Neena|17000|2|
|90|102|Lex|17000|2|
|100|108|Nancy|12008|1|
|100|109|Daniel|9000|2|
|100|110|John|8200|3|
|110|205|Shelley|12008|1|
|110|206|William|8300|2|
||178|Kimberely|7000|1|


INNER JOIN - show employees with managers

```sql
select  e.EMPLOYEE_ID ,  e.FIRST_NAME , e.manager_id , m.first_name as manager_name
from hr.employees e
inner join hr.employees m on m.employee_id = e.manager_id 
```

|EMPLOYEE_ID|FIRST_NAME|MANAGER_ID|MANAGER_NAME|
|--|--|--|--|
|101|Neena|100|Steven|
|102|Lex|100|Steven|
|103|Alexander|102|Lex|
|104|Bruce|103|Alexander|
|105|David|103|Alexander|
|106|Valli|103|Alexander|
|107|Diana|103|Alexander|
|108|Nancy|101|Neena|
|109|Daniel|108|Nancy|
|110|John|108|Nancy|

LEFT JOIN - show employees with or without managers 

```sql
select  e.EMPLOYEE_ID ,  e.FIRST_NAME , e.manager_id , m.first_name as manager_name
from hr.employees e
left join hr.employees m on m.employee_id = e.manager_id 
order by e.employee_id 
Fetch first 10 rows only
```

|EMPLOYEE_ID|FIRST_NAME|MANAGER_ID|MANAGER_NAME|
|--|--|--|--|
|100|Steven|||
|101|Neena|100|Steven|
|102|Lex|100|Steven|
|103|Alexander|102|Lex|
|104|Bruce|103|Alexander|
|105|David|103|Alexander|
|106|Valli|103|Alexander|
|107|Diana|103|Alexander|
|108|Nancy|101|Neena|
|109|Daniel|108|Nancy|

RIGHT JOIN - show managers with or without employees

```sql
select  e.EMPLOYEE_ID ,  e.FIRST_NAME ,m.EMPLOYEE_ID as manager_id, m.first_name as manager_name
from hr.employees e
right join hr.employees m on m.employee_id = e.manager_id 
order by m.employee_id desc
fetch first 10 rows only
 
```

|EMPLOYEE_ID|FIRST_NAME|MANAGER_ID|MANAGER_NAME|
|--|--|--|--|
|||206|William|
|206|William|205|Shelley|
|||204|Hermann|
|||203|Susan|
|||202|Pat|
|202|Pat|201|Michael|
|||200|Jennifer|
|||199|Douglas|
|||198|Donald|
|||197|Kevin|


FULL OUTER JOIN - show employees with or without managers also show managers with or without employees

```sql

select  e.EMPLOYEE_ID ,  e.FIRST_NAME ,m.EMPLOYEE_ID as manager_id, m.first_name as manager_name
from hr.employees e
full outer join hr.employees m on m.employee_id = e.manager_id 
```

|EMPLOYEE_ID|FIRST_NAME|MANAGER_ID|MANAGER_NAME|
|--|--|--|--|
|101|Neena|100|Steven|
|102|Lex|100|Steven|
|104|Bruce|103|Alexander|
|105|David|103|Alexander|
|106|Valli|103|Alexander|
|107|Diana|103|Alexander|
|||104|Bruce|
|||105|David|
|||106|Valli|
|||107|Diana|
|100|Steven|||


Show employee hierarchy 
```sql
select level, e.employee_id , e.manager_id
from hr.employees e 
start with e.manager_id is null 
connect by prior e.employee_id = e.manager_id 
 order by level
```


|LEVEL|EMPLOYEE_ID|MANAGER_ID|
|--|--|--|
|1|100||
|2|102|100|
|2|114|100|
|2|120|100|
|2|121|100|
|2|122|100|
|2|123|100|
|2|124|100|
|2|145|100|
|2|146|100|
|2|147|100|
|2|148|100|
|2|149|100|
|2|201|100|
|2|101|100|
|3|108|101|
|3|200|101|
|3|203|101|
|3|204|101|
|3|205|101|
|3|103|102|
|3|115|114|
|3|116|114|
|3|117|114|
|3|118|114|
|3|119|114|
|3|125|120|
|3|126|120|
|3|127|120|
|3|128|120|
|3|180|120|
|3|181|120|
|3|182|120|
|3|183|120|
|3|129|121|
|3|130|121|
|3|131|121|
|3|132|121|
|3|184|121|
|3|185|121|
|3|186|121|
|3|187|121|
|3|133|122|
|3|134|122|
|3|135|122|
|3|136|122|
|3|188|122|
|3|189|122|
|3|190|122|
|3|191|122|
|3|137|123|
|3|138|123|
|3|139|123|
|3|140|123|
|3|192|123|
|3|193|123|
|3|194|123|
|3|195|123|
|3|141|124|
|3|142|124|
|3|143|124|
|3|144|124|
|3|196|124|
|3|197|124|
|3|198|124|
|3|199|124|
|3|150|145|
|3|151|145|
|3|152|145|
|3|153|145|
|3|154|145|
|3|155|145|
|3|156|146|
|3|157|146|
|3|158|146|
|3|159|146|
|3|160|146|
|3|161|146|
|3|162|147|
|3|163|147|
|3|164|147|
|3|165|147|
|3|166|147|
|3|167|147|
|3|168|148|
|3|169|148|
|3|170|148|
|3|171|148|
|3|172|148|
|3|173|148|
|3|174|149|
|3|175|149|
|3|176|149|
|3|177|149|
|3|178|149|
|3|179|149|
|3|202|201|
|4|109|108|
|4|110|108|
|4|111|108|
|4|112|108|
|4|107|103|
|4|206|205|
|4|104|103|
|4|105|103|
|4|106|103|
|4|113|108|

 
