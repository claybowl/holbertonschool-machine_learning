-- adds name and country to table users
--
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,  -- id: integer, never null, auto increment, primary key
    email VARCHAR(255) NOT NULL UNIQUE,  -- email: string (255 characters), never null, unique
    name VARCHAR(255),                   -- name: string (255 characters)
    country ENUM('US', 'CO', 'TN') NOT NULL DEFAULT 'US'  -- country: enumeration of countries, never null, default 'US'
);
