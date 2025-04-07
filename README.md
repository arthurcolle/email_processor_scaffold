# Email Processor Scaffold

This is the scaffolded project for candidates to implement email processing logic.

### Running with Docker Compose

1. Build and start the services:
   ```bash
   docker compose up
   ```
2. The application will be available at `http://localhost:8000`
3. To confirm the app is running, send a POST request to `http://localhost:8000/webhook`. You should see a response of `{status: ok}`
