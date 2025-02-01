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

