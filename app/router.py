import importlib
import os
import glob
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

def register_routes(app: FastAPI):
    """
    Dynamically register all routes from the routes directory.
    """
    routes_dir = os.path.join(os.path.dirname(__file__), "routes")
    
    # Check if the routes directory exists
    if not os.path.exists(routes_dir):
        logger.warning(f"Routes directory {routes_dir} does not exist")
        return
        
    # Find all Python files in the routes directory
    route_files = glob.glob(os.path.join(routes_dir, "*.py"))
    
    # Setup static files and templates
    static_dir = Path(__file__).parent / "static"
    templates_dir = Path(__file__).parent / "templates"
    
    # Ensure directories exist
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(templates_dir, exist_ok=True)
    
    # Mount static files
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Register all route modules
    for route_file in route_files:
        # Skip __init__.py and other special files
        if os.path.basename(route_file).startswith('__'):
            continue
            
        # Convert file path to module path
        # Example: ./app/routes/email_routes.py -> app.routes.email_routes
        relative_path = os.path.relpath(route_file, os.path.dirname(os.path.dirname(routes_dir)))
        module_path = os.path.splitext(relative_path)[0].replace(os.sep, ".")
        
        # Special case for email_routes_sql.py - use this instead of email_routes.py
        if os.path.basename(route_file) == "email_routes_sql.py":
            # Skip the original email_routes.py as we'll use the SQL version
            continue
            
        try:
            # Import the module dynamically
            route_module = importlib.import_module(module_path)
            
            # Check if the module has a router attribute
            if hasattr(route_module, "router"):
                app.include_router(route_module.router)
                logger.info(f"Registered routes from {module_path}")
            else:
                logger.warning(f"Module {module_path} does not have a router attribute")
        except ImportError as e:
            logger.error(f"Failed to import {module_path}: {e}")
            
    # Import SQL email routes specifically
    try:
        from app.routes.email_routes_sql import router as email_sql_router
        app.include_router(email_sql_router)
        logger.info("Registered routes from app.routes.email_routes_sql")
    except ImportError as e:
        logger.error(f"Failed to import app.routes.email_routes_sql: {e}")