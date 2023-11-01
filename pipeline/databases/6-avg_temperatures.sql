-- Calculate the average temperature by city
--
SELECT city, AVG(temperature) AS avg_temp
FROM temperatures  -- Assuming the table name is 'temperatures'
GROUP BY city
ORDER BY avg_temp DESC;
