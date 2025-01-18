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

