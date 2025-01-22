### [2356. Number of Unique Subjects Taught by Each Teacher](https://leetcode.cn/problems/number-of-unique-subjects-taught-by-each-teacher/)

```mysql
SELECT
    teacher_id, COUNT(DISTINCT subject_id) AS cnt 
FROM
    Teacher 
GROUP BY
    teacher_id
```

