import os
import subprocess

def run_script(script_name):
    print(f"\nEjecutando {script_name}...\n")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error ejecutando {script_name}:\n{result.stderr}")
        exit(1)

def main():
    # Paso 1: Descargar datos
    run_script("scripts/download_multi_timeframes.py")

    # Paso 2: Combinar datos
    run_script("scripts/combine_timeframes.py")

    # Paso 3: Entrenar modelo robusto
    run_script("scripts/train_robust_model.py")

if __name__ == "__main__":
    main()
