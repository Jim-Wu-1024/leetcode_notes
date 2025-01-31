### [1667. Fix Names in a Table](https://leetcode.cn/problems/fix-names-in-a-table/)

```mysql
SELECT
    user_id, CONCAT(UPPER(LEFT(name, 1)), LOWER(RIGHT(name, LENGTH(name) - 1))) AS name
FROM
    Users
ORDER BY
    user_id
```

