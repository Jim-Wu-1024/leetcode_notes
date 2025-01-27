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

### [180. Consecutive Numbers](https://leetcode.cn/problems/consecutive-numbers/)

```mysql
SELECT
    DISTINCT l1.num AS ConsecutiveNums
FROM
    Logs l1, Logs l2, Logs l3
WHERE
    l1.id = l2.id - 1 AND l2.id = l3.id - 1 AND l1.num = l2.num AND l2.num = l3.num
```

### [1164. Product Price at a Given Date](https://leetcode.cn/problems/product-price-at-a-given-date/)

```mysql
SELECT
    t1.product_id, IFNULL(t2.new_price, 10) AS price
FROM
(
SELECT
    DISTINCT product_id
FROM
    Products
) t1
LEFT JOIN
(
SELECT
    product_id, new_price
FROM
    Products
WHERE
    (product_id, change_date)
IN 
    (
    SELECT
        product_id, MAX(change_date)
    FROM
        Products
    WHERE
        change_date <= "2019-08-16"
    GROUP BY
        product_id
    )
) t2
ON
    t1.product_id = t2.product_id
```

### [1204. Last Person to Fit in the Bus](https://leetcode.cn/problems/last-person-to-fit-in-the-bus/)

using OLAP to calculate the accumulation by `SUM()`.

```mysql
SELECT
    t.person_name
FROM
(
SELECT
    person_name, SUM(weight) OVER (ORDER BY turn) AS cumu_weight
FROM
    Queue
) t
WHERE
    cumu_weight <= 1000
ORDER BY
    cumu_weight DESC
LIMIT 1
```

