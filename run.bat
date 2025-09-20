@echo off
echo Starting server and client nodes...

:: Open a new Command Prompt for the server
start cmd /k python server.py

:: Wait briefly to ensure server starts first
timeout /t 2 /nobreak

:: Open a new Command Prompt for clients
start cmd /k python client.py --node-id=0

start cmd /k python client.py --node-id=1

start cmd /k python client.py --node-id=2

start cmd /k python client.py --node-id=3

start cmd /k python client.py --node-id=4

start cmd /k python client.py --node-id=5

start cmd /k python client.py --node-id=6


echo All nodes started.
exit