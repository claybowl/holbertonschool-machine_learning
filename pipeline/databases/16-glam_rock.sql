-- list all bands with glam rock as their main style
--
SELECT band_name, 
       CASE 
           WHEN split IS NOT NULL THEN split - formed
           ELSE 2020 - formed
       END AS 'lifespan until 2020 (in years)'
FROM metal_bands
WHERE main_style = 'Glam rock'
ORDER BY `lifespan until 2020 (in years)` DESC;
