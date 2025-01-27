### [1731. The Number of Employees Which Report to Each Employee](https://leetcode.cn/problems/the-number-of-employees-which-report-to-each-employee/)

```mysql
SELECT
    e2.employee_id AS employee_id, e2.name AS name, COUNT(e2.employee_id) AS reports_count, ROUND(AVG(e1.age), 0) AS average_age
FROM 
    Employees e1
LEFT JOIN
    Employees e2
ON
    e1.reports_to = e2.employee_id
WHERE
    e2.employee_id IS NOT NULL
GROUP BY
    e2.employee_id
ORDER BY
    e2.employee_id
```

### [1789. Primary Department for Each Employee](https://leetcode.cn/problems/primary-department-for-each-employee/)

```mysql
SELECT
    employee_id, department_id
FROM 
    Employee 
WHERE
    primary_flag = "Y"
UNION
SELECT
    employee_id, department_id
FROM    
    Employee
GROUP BY
    employee_id
HAVING
    COUNT(department_id) = 1
```

### [610. Triangle Judgement](https://leetcode.cn/problems/triangle-judgement/)

```mysql
SELECT
    x, y, z, IF((x + y > z AND x + z > y AND y + z > x), "Yes", "No") AS triangle
FROM
    Triangle
```

