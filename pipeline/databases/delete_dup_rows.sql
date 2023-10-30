-- deletes duplicate rows by using temporary table
--
CREATE TEMPORARY TABLE temp_table AS SELECT DISTINCT * FROM second_table;
TRUNCATE TABLE second_table;
INSERT INTO second_table SELECT * FROM temp_table;
DROP TEMPORARY TABLE temp_table;
