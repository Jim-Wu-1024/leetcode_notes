### [620. Not Boring Movies](https://leetcode.cn/problems/not-boring-movies/)

**find_in_set("", )** to determine whether the string contains specific word.

```mysql
SELECT
    *
FROM
    Cinema
WHERE
    id % 2 = 1 AND !find_in_set('boring', description)
ORDER BY
    rating DESC
```

### [1251. Average Selling Price](https://leetcode.cn/problems/average-selling-price/)

```mysql
SELECT
    id as product_id, ROUND(IFNULL(SUM(sales) / SUM(units), 0), 2) as average_price
FROM
(SELECT
    p.product_id AS id, p.price * u.units as sales, u.units AS units
FROM
    Prices p 
LEFT JOIN
    UnitsSold u 
ON 
    p.product_id = u.product_id AND (u.purchase_date BETWEEN p.start_date AND p.end_date)
) t
GROUP BY
    product_id
```

### [1075. Project Employees I](https://leetcode.cn/problems/project-employees-i/)

```mysql
SELECT
    p.project_id, ROUND(AVG(e.experience_years), 2) AS average_years
FROM
    Project p
LEFT JOIN
    Employee e 
ON
    p.employee_id = e.employee_id
GROUP BY
    p.project_id

```

### [1633. Percentage of Users Attended a Contest](https://leetcode.cn/problems/percentage-of-users-attended-a-contest/)

```mysql
SELECT
    r.contest_id as contest_id, ROUND(COUNT(u1.user_id) / ct.cnt, 4) * 100 AS percentage
FROM
    Register r
LEFT JOIN
    Users u1    
ON
    r.user_id = u1.user_id
JOIN
(SELECT
    COUNT(*) as cnt
FROM 
    Users u2
) ct
GROUP BY
    r.contest_id
ORDER BY
    percentage DESC,
    r.contest_id ASC
```

### [1211. Queries Quality and Percentage](https://leetcode.cn/problems/queries-quality-and-percentage/)

```mysql
SELECT
    t1.query_name, ROUND(AVG(t2.quality), 2) AS quality, ROUND(SUM(t2.rating_quality) / COUNT(t2.rating_quality), 4) * 100 AS poor_query_percentage
FROM
    Queries t1
LEFT JOIN
(
    SELECT  
        query_name, IF((rating < 3), 1, 0) AS rating_quality, rating / position AS quality
    FROM
        Queries
) t2
ON
    t1.query_name = t2.query_name
GROUP BY
    t1.query_name
```

### [1193. Monthly Transactions I](https://leetcode.cn/problems/monthly-transactions-i/)

```mysql
SELECT
    DATE_FORMAT(trans_date, '%Y-%m') AS month,
    country,
    COUNT(state) AS trans_count,
    SUM(IF((state = 'approved'), 1, 0)) AS approved_count,
    SUM(amount) AS trans_total_amount,
    SUM(IF((state = 'approved'), amount, 0)) AS approved_total_amount
FROM
    Transactions
GROUP BY
    month, country
```

### [1174. Immediate Food Delivery II](https://leetcode.cn/problems/immediate-food-delivery-ii/)

```mysql
SELECT
    ROUND(SUM(order_date = customer_pref_delivery_date) / COUNT(*), 4) * 100 AS immediate_percentage
FROM
    Delivery
WHERE
    (customer_id, order_date)
IN
(
    SELECT 
        customer_id, MIN(order_date)
    FROM
        Delivery
    GROUP BY
        customer_id
)
```

### [550. Game Play Analysis IV](https://leetcode.cn/problems/game-play-analysis-iv/)

```mysql
SELECT 
    ROUND(SUM(IF(DATEDIFF(event_date, first_date)=1, 1, 0)) / COUNT(DISTINCT a1.player_id), 2) AS fraction
FROM
    Activity a1
LEFT JOIN
(
    SELECT
        player_id, MIN(event_date) AS first_date
    FROM
        Activity
    GROUP BY
        player_id
) a2
ON
    a1.player_id = a2.player_id
```

