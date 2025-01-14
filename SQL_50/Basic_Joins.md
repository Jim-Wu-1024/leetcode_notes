### [1378. Replace Employee ID With The Unique Identifier](https://leetcode.cn/problems/replace-employee-id-with-the-unique-identifier/)

```sql
SELECT
    EmployeeUNI.unique_id, Employees.name
FROM
    Employees
LEFT JOIN
    EmployeeUNI
ON
    EmployeeUNI.id = Employees.id
```

