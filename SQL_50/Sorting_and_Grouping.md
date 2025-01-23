### [2356. Number of Unique Subjects Taught by Each Teacher](https://leetcode.cn/problems/number-of-unique-subjects-taught-by-each-teacher/)

```mysql
SELECT
    teacher_id, COUNT(DISTINCT subject_id) AS cnt 
FROM
    Teacher 
GROUP BY
    teacher_id
```

### [1141. User Activity for the Past 30 Days I](https://leetcode.cn/problems/user-activity-for-the-past-30-days-i/)

```mysql
SELECT
    activity_date AS day, COUNT(DISTINCT user_id) AS active_users
FROM
    Activity
WHERE
    activity_date BETWEEN '2019-06-28' AND '2019-07-27'
GROUP BY
    activity_date
```

### [1084. Sales Analysis III](https://leetcode.cn/problems/sales-analysis-iii/)

```mysql
SELECT
    p.product_id AS product_id, p.product_name AS product_name
FROM
    Sales s
LEFT JOIN
    Product p
ON
    s.product_id = p.product_id
GROUP BY
    p.product_id
HAVING
    MIN(s.sale_date) >= '2019-01-01' AND MAX(s.sale_date) <= '2019-03-31'
```

### [596. Classes More Than 5 Students](https://leetcode.cn/problems/classes-more-than-5-students/)

```mysql
SELECT
    class
FROM
    Courses
GROUP BY
    class
HAVING
    COUNT(student) >= 5
```

