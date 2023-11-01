-- list all bands with glam rock as their main style
--
SELECT band_name, IFNULL((split-formed), (2020 - formed)) AS lifespan
FROM metal_bands
WHERE style LIKE '%Glam rock%'
ORDER BY lifespan DESC;
