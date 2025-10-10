@echo off
:: ============================================================
::  GW-MF-PINN Project Bootstrap Script
::  Cria toda a estrutura de diretórios e arquivos básicos
::  Autor: Wallace Ferreira
:: ============================================================

set "ROOT=gw-mf-pinn"

echo ========================================
echo Criando estrutura do projeto %ROOT% ...
echo ========================================

:: ---------- Pastas principais ----------
mkdir %ROOT%
mkdir %ROOT%\configs
mkdir %ROOT%\data
mkdir %ROOT%\data\raw
mkdir %ROOT%\data\interim
mkdir %ROOT%\data\processed
mkdir %ROOT%\notebooks
mkdir %ROOT%\src
mkdir %ROOT%\src\gwdata
mkdir %ROOT%\src\sim
mkdir %ROOT%\src\models
mkdir %ROOT%\src\training
mkdir %ROOT%\src\eval
mkdir %ROOT%\src\viz
mkdir %ROOT%\tests

:: ---------- Arquivos de configuração ----------
echo # Projeto GW-MF-PINN: Arquitetura Multi-Fidelity Physics-Informed >> %ROOT%\README.md
echo Descrição inicial do repositório de pesquisa. >> %ROOT%\README.md

echo # Configurações de Dados >> %ROOT%\configs\data.yaml
echo root_path: ./data/raw >> %ROOT%\configs\data.yaml

echo # Hiperparâmetros PINN >> %ROOT%\configs\train_pinn.yaml
echo epochs: 10000 >> %ROOT%\configs\train_pinn.yaml
echo learning_rate: 1e-4 >> %ROOT%\configs\train_pinn.yaml

echo # Hiperparâmetros MF-PINN >> %ROOT%\configs\train_mf.yaml
echo alpha: 0.3 >> %ROOT%\configs\train_mf.yaml
echo beta: 0.5 >> %ROOT%\configs\train_mf.yaml
echo gamma: 1.0 >> %ROOT%\configs\train_mf.yaml

echo # CNN baseline hyperparams >> %ROOT%\configs\baseline_cnn.yaml
echo filters: [32,64,128] >> %ROOT%\configs\baseline_cnn.yaml

:: ---------- Arquivos fonte ----------
echo # Download e preprocessamento de dados do LIGO >> %ROOT%\src\gwdata\gwosc.py
echo # Simulações pós-newtonianas e SEOBNR >> %ROOT%\src\sim\templates.py
echo # Formas analíticas de baixa fidelidade >> %ROOT%\src\sim\post_newtonian.py
echo # Modelo PINN >> %ROOT%\src\models\pinn.py
echo # Modelo Multi-Fidelity PINN >> %ROOT%\src\models\mf_pinn.py
echo # Funções de perda físicas >> %ROOT%\src\models\losses.py
echo # Baselines (matched filter e CNN) >> %ROOT%\src\models\baselines.py
echo # Loop de treino PINN >> %ROOT%\src\training\train_pinn.py
echo # Loop de treino MF-PINN >> %ROOT%\src\training\train_mf.py
echo # Avaliação de métricas >> %ROOT%\src\eval\metrics.py
echo # Comparação entre modelos >> %ROOT%\src\eval\compare.py
echo # Visualizações (tempo-frequência, Q-transform) >> %ROOT%\src\viz\timefreq.py
echo # Plotagem de resultados e curvas >> %ROOT%\src\viz\plots.py

:: ---------- Arquivos utilitários ----------
echo requests==2.31.0 > %ROOT%\requirements.txt
echo numpy==1.26.4 >> %ROOT%\requirements.txt
echo scipy==1.13.0 >> %ROOT%\requirements.txt
echo torch==2.2.2 >> %ROOT%\requirements.txt
echo matplotlib==3.9.0 >> %ROOT%\requirements.txt
echo pandas==2.2.2 >> %ROOT%\requirements.txt
echo scikit-learn==1.5.1 >> %ROOT%\requirements.txt
echo pycbc==2.2.3 >> %ROOT%\requirements.txt
echo jupyterlab==4.2.1 >> %ROOT%\requirements.txt

echo # Ambiente virtual exemplo > %ROOT%\.env.example
echo DATA_PATH=./data/raw >> %ROOT%\.env.example
echo OUTPUT_PATH=./data/processed >> %ROOT%\.env.example

echo all: setup >> %ROOT%\Makefile
echo setup: >> %ROOT%\Makefile
echo ^tpython -m venv .venv >> %ROOT%\Makefile
echo ^tpip install -r requirements.txt >> %ROOT%\Makefile

:: ---------- Testes ----------
echo # Teste básico de importação >> %ROOT%\tests\test_imports.py
echo import os, sys >> %ROOT%\tests\test_imports.py
echo print("Estrutura inicial carregada com sucesso.") >> %ROOT%\tests\test_imports.py

echo.
echo ========================================
echo Estrutura criada com sucesso!
echo Diretório: %CD%\%ROOT%
echo ========================================
pause