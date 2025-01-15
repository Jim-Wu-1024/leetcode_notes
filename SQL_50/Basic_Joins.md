### [1378. Replace Employee ID With The Unique Identifier](https://leetcode.cn/problems/replace-employee-id-with-the-unique-identifier/)

```mysql
SELECT
    EmployeeUNI.unique_id, Employees.name
FROM
    Employees
LEFT JOIN
    EmployeeUNI
ON
    EmployeeUNI.id = Employees.id
```

### [1068. Product Sales Analysis I](https://leetcode.cn/problems/product-sales-analysis-i/)

```mysql
SELECT 
    Product.product_name, Sales.year, Sales.price
FROM
    Sales
LEFT JOIN
    Product
ON
    Sales.product_id = Product.product_id
```

### [1581. Customer Who Visited but Did Not Make Any Transactions](https://leetcode.cn/problems/customer-who-visited-but-did-not-make-any-transactions/)

```mysql
SELECT
    Visits.customer_id, COUNT(Visits.customer_id) as count_no_trans
FROM
    Visits
LEFT JOIN
    Transactions
ON
    Visits.visit_id = Transactions.visit_id
WHERE
    ISNULL(Transactions.transaction_id)
GROUP BY
    Visits.customer_id
```

The question is to find customers who do not trade. If amount is null, it means that the customer does not consume. It is also possible that the customer has traded but not consumed. This does not meet the requirement of no trade, so an error will occur. Therefore, amount is null is not acceptable.

### [197. Rising Temperature](https://leetcode.cn/problems/rising-temperature/)

```mysql
SELECT
    w2.id
FROM 
    Weather w1, Weather w2
WHERE
    datediff(w2.recordDate, w1.recordDate) = 1 AND w2.temperature > w1.temperature
```

