### [1667. Fix Names in a Table](https://leetcode.cn/problems/fix-names-in-a-table/)

```mysql
SELECT
    user_id, CONCAT(UPPER(LEFT(name, 1)), LOWER(RIGHT(name, LENGTH(name) - 1))) AS name
FROM
    Users
ORDER BY
    user_id
```

### [1527. Patients With a Condition](https://leetcode.cn/problems/patients-with-a-condition/)

```mysql
SELECT
    patient_id, patient_name, conditions
FROM
    Patients
WHERE
    conditions LIKE "DIAB1%" OR conditions LIKE "% DIAB1%"
```

### [196. Delete Duplicate Emails](https://leetcode.cn/problems/delete-duplicate-emails/)

```mysql
DELETE FROM Person
WHERE
    id IN
    (
        SELECT 
            id 
        FROM
        (
            SELECT
                id, ROW_NUMBER() OVER (PARTITION BY email ORDER BY id) as rn
            FROM
                Person
        ) a
        WHERE
            a.rn > 1
    )
```

### [176. Second Highest Salary](https://leetcode.cn/problems/second-highest-salary/)

```mysql
SELECT 
    IFNULL((SELECT DISTINCT salary FROM Employee 
            ORDER BY Salary Desc
            LIMIT 1 OFFSET 1), NULL
           ) AS SecondHighestSalary
```

### [1484. Group Sold Products By The Date](https://leetcode.cn/problems/group-sold-products-by-the-date/)

```mysql
SELECT
    sell_date, COUNT(DISTINCT product) AS num_sold, GROUP_CONCAT(DISTINCT product ORDER BY product SEPARATOR ',') AS products
FROM
    Activities
GROUP BY
    sell_date
ORDER BY    
    sell_date
```

### [1327. List the Products Ordered in a Period](https://leetcode.cn/problems/list-the-products-ordered-in-a-period/)

```mysql
SELECT
    p.product_name, SUM(o.unit) AS unit
FROM
    Products p
LEFT JOIN
    Orders o
ON
    p.product_id = o.product_id
WHERE
    o.order_date BETWEEN "2020-02-01" AND "2020-02-29"
GROUP BY
    product_name
HAVING 
    unit >= 100
```

### [1517. Find Users With Valid E-Mails](https://leetcode.cn/problems/find-users-with-valid-e-mails/)

```mysql
SELECT
    user_id, name, mail
FROM
    Users
WHERE
    mail REGEXP '^[a-zA-Z][a-zA-Z0-9_.-]*\\@leetcode\\.com$'
```

