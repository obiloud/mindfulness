CREATE EXTENSION IF NOT EXISTS vector;

-- Create users table
-- This table stores user credentials and metadata
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create messages table
-- Stores full conversation history per thread, linked to user
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY,
    thread_id TEXT NOT NULL,
    user_id UUID NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'ai')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_thread_user ON messages (thread_id, user_id);
