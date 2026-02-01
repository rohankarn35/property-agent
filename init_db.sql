-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Schools reference table
CREATE TABLE schools (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    geom GEOGRAPHY(POINT, 4326) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Parcels/Properties table
CREATE TABLE parcels (
    parcel_id VARCHAR(50) PRIMARY KEY,
    address TEXT NOT NULL,
    area_sqft DECIMAL(10,2) NOT NULL,  -- Living area in square feet
    property_type VARCHAR(50),         -- 'residential', 'commercial'
    geom GEOGRAPHY(POINT, 4326) NOT NULL  -- Parcel centroid
);

-- Critical Indexes
CREATE INDEX idx_schools_geom ON schools USING GIST(geom);
CREATE INDEX idx_schools_name ON schools (name);
CREATE INDEX idx_schools_name_trgm ON schools USING GIN (name gin_trgm_ops);  -- For fuzzy matching
CREATE INDEX idx_parcels_geom ON parcels USING GIST(geom);
CREATE INDEX idx_parcels_area ON parcels(area_sqft);

-- Set trigram similarity threshold
SET pg_trgm.similarity_threshold = 0.3;
