all: setup 
setup: 
tpython -m venv .venv 
tpip install -r requirements.txt 
