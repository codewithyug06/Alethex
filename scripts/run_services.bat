@echo off
title ALETHEX Memory Gateway & Universal RAG Services
echo ========================================================
echo   Starting ALETHEX Universal AI Memory Services
echo   - Universal OpenAI Proxy on: http://localhost:8000
echo   - Universal RAG Microservice on: http://localhost:8080
echo ========================================================
python -m alethex.integrations.rag_service --port 8080
pause
