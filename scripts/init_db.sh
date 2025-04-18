#!/bin/bash
set -e

cd /app

echo "Initializing database..."
python init_db.py

echo "Database initialization complete!"