### 1757. Recyclable and Low Fat Products

```sql
SELECT 
    product_id 
FROM 
    Products 
WHERE
    low_fats = "Y" AND recyclable = "Y"
```

### 584. Find Customer Referee

```sql
SELECT
    name
FROM
    Customer
WHERE
    referee_id <> 2 OR referee_id IS NULL
```

### 595. Big Countries

```sql
SELECT
    name, population, area
FROM
    World
WHERE
    area >= 3000000 OR population >= 25000000
```

