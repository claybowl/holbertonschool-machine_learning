-- Create a function to safely divide two numbers
--
DELIMITER //
CREATE FUNCTION SafeDiv(a INT, b INT) RETURNS FLOAT
BEGIN
    DECLARE result FLOAT;
    
    -- Check if the divisor is zero
    IF b = 0 THEN
        SET result = 0;
    ELSE
        SET result = a / b;
    END IF;
    
    RETURN result;
END;
//
DELIMITER ;
