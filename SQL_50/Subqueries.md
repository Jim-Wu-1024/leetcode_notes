### [1978. Employees Whose Manager Left the Company](https://leetcode.cn/problems/employees-whose-manager-left-the-company/)

```mysql
SELECT
    employee_id
FROM
    Employees
WHERE 
    salary < 30000 AND manager_id NOT IN
    (
        SELECT
            employee_id
        FROM
            Employees
    )
ORDER BY
    employee_id
```

### [626. Exchange Seats](https://leetcode.cn/problems/exchange-seats/)

```mysql
SELECT
    RANK() OVER(order by IF(id MOD 2 = 0, id - 2, id)) AS id, student
FROM
    Seat
```

