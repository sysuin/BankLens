-- Runs once when the Compose Postgres volume is first created.
-- The API connects as banklens_app (subject to row-level security);
-- migrations and the seed run as the owner (POSTGRES_USER).
CREATE ROLE banklens_app LOGIN PASSWORD 'banklens_app';
GRANT CONNECT ON DATABASE banklens TO banklens_app;
-- The chat's warehouse role: SELECT on tenant-filtered views only
-- (grants are made by migration 0006).
CREATE ROLE banklens_chat LOGIN NOINHERIT PASSWORD 'banklens_chat';
GRANT CONNECT ON DATABASE banklens TO banklens_chat;
