### [1757. Recyclable and Low Fat Products](https://leetcode.cn/problems/recyclable-and-low-fat-products/)

```mysql
SELECT 
    product_id 
FROM 
    Products 
WHERE
    low_fats = "Y" AND recyclable = "Y"
```

### [584. Find Customer Referee](https://leetcode.cn/problems/find-customer-referee/)

```mysql
SELECT
    name
FROM
    Customer
WHERE
    referee_id <> 2 OR referee_id IS NULL
```

### [595. Big Countries](https://leetcode.cn/problems/big-countries/)

```mysql
SELECT
    name, population, area
FROM
    World
WHERE
    area >= 3000000 OR population >= 25000000
```

### [1148. Article Views I](https://leetcode.cn/problems/article-views-i/)

```mysql
SELECT
    DISTINCT author_id AS id 
FROM
    Views
WHERE 
    author_id = viewer_id
ORDER BY
    id ASC
```

### [1683. Invalid Tweets](https://leetcode.cn/problems/invalid-tweets/)

```mysql
SELECT
    tweet_id
FROM
    Tweets
WHERE
    length(content) > 15
```

It is better to use CHAR_LENGTH() instead of LENGTH() because the length of some special characters is longer than 1 byte.
