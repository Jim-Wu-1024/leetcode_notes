### [1378. Replace Employee ID With The Unique Identifier](https://leetcode.cn/problems/replace-employee-id-with-the-unique-identifier/)

```mysql
SELECT
    EmployeeUNI.unique_id, Employees.name
FROM
    Employees
LEFT JOIN
    EmployeeUNI
ON
    EmployeeUNI.id = Employees.id
```

### [1068. Product Sales Analysis I](https://leetcode.cn/problems/product-sales-analysis-i/)

```mysql
SELECT 
    Product.product_name, Sales.year, Sales.price
FROM
    Sales
LEFT JOIN
    Product
ON
    Sales.product_id = Product.product_id
```

### [1581. Customer Who Visited but Did Not Make Any Transactions](https://leetcode.cn/problems/customer-who-visited-but-did-not-make-any-transactions/)

```mysql
SELECT
    Visits.customer_id, COUNT(Visits.customer_id) as count_no_trans
FROM
    Visits
LEFT JOIN
    Transactions
ON
    Visits.visit_id = Transactions.visit_id
WHERE
    ISNULL(Transactions.transaction_id)
GROUP BY
    Visits.customer_id
```

The question is to find customers who do not trade. If amount is null, it means that the customer does not consume. It is also possible that the customer has traded but not consumed. This does not meet the requirement of no trade, so an error will occur. Therefore, amount is null is not acceptable.

### [197. Rising Temperature](https://leetcode.cn/problems/rising-temperature/)

```mysql
SELECT
    w2.id
FROM 
    Weather w1, Weather w2
WHERE
    datediff(w2.recordDate, w1.recordDate) = 1 AND w2.temperature > w1.temperature
```

### [1661. Average Time of Process per Machine](https://leetcode.cn/problems/average-time-of-process-per-machine/)

```mysql
SELECT
    a1.machine_id, ROUND(AVG(a2.timestamp - a1.timestamp), 3) AS processing_time
FROM
    Activity a1, Activity a2
WHERE
    a1.machine_id = a2.machine_id AND a1.process_id = a2.process_id AND a1.activity_type = "start" AND a2.activity_type = "end"
GROUP BY
    a1.machine_id
```

### [577. Employee Bonus](https://leetcode.cn/problems/employee-bonus/)

```mysql
SELECT
    e.name, b.bonus
FROM
    Employee e
LEFT JOIN
    Bonus b
ON
    e.empID = b.empID
WHERE
    b.bonus < 1000 OR b.bonus IS NULL
```

### [1280. Students and Examinations](https://leetcode.cn/problems/students-and-examinations/)

```mysql
SELECT
        stu.student_id, stu.student_name, sub.subject_name, COUNT(e.subject_name) AS attended_exams
FROM
    Students stu 
CROSS JOIN
    Subjects sub 
LEFT JOIN
    Examinations e
ON
    stu.student_id = e.student_id AND sub.subject_name = e.subject_name
GROUP BY
    stu.student_id, sub.subject_name
ORDER BY
    stu.student_id, sub.subject_name
```

### [570. Managers with at Least 5 Direct Reports](https://leetcode.cn/problems/managers-with-at-least-5-direct-reports/)

```mysql
SELECT
    name
FROM
    Employee employee
INNER JOIN
(SELECT
    managerId
FROM
    Employee
GROUP BY
    managerId
HAVING
    COUNT(managerId) >= 5
) AS manager
ON
    employee.id = manager.managerId
```

### [1934. Confirmation Rate](https://leetcode.cn/problems/confirmation-rate/)

```mysql
SELECT 
    s.user_id, ROUND(IFNULL(AVG(c.action = "confirmed"), 0), 2) AS confirmation_rate
FROM
    Signups s
LEFT JOIN
    Confirmations c
ON
    s.user_id = c.user_id
GROUP BY
    s.user_id
```

