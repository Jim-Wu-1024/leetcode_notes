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

### [1341. Movie Rating](https://leetcode.cn/problems/movie-rating/)

```mysql
(SELECT 
    name AS results
FROM
    Users u
LEFT JOIN
    (
    SELECT 
        user_id, COUNT(user_id) AS cnt
    FROM 
        MovieRating
    GROUP BY
        user_id
    ) t1
ON
    u.user_id = t1.user_id
ORDER BY
    t1.cnt DESC,
    u.name ASC
LIMIT 1)
UNION ALL
(SELECT 
    title AS results
FROM
    Movies m 
LEFT JOIN
(
SELECT 
    movie_id, AVG(rating) AS max_rating
FROM
    MovieRating
WHERE
    created_at BETWEEN "2020-02-01" AND "2020-02-28"
GROUP BY
    movie_id
) t2
ON m.movie_id = t2.movie_id
ORDER BY
    t2.max_rating DESC,
    m.title ASC
LIMIT 1)
```

### [1321. Restaurant Growth](https://leetcode.cn/problems/restaurant-growth/)

```mysql
SELECT
    DISTINCT t.visited_on, t.sum_amount AS amount, ROUND(t.sum_amount / 7, 2) AS average_amount
FROM
(
SELECT
    visited_on, SUM(amount) OVER (ORDER BY visited_on RANGE INTERVAL 6 DAY PRECEDING) AS sum_amount
FROM
    Customer
) t
WHERE
    visited_on >= (SELECT MIN(visited_on) FROM Customer) + 6
```

### [602. Friend Requests II: Who Has the Most Friends](https://leetcode.cn/problems/friend-requests-ii-who-has-the-most-friends/)

```mysql
SELECT
    t1.ids AS id, COUNT(ids) AS num
FROM
(
    SELECT requester_id as ids FROM RequestAccepted
    UNION ALL
    SELECT accepter_id as ids FROM RequestAccepted
) t1
GROUP BY
    t1.ids
ORDER BY
    num DESC
LIMIT 1
```

