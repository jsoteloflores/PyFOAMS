# run.ps1

param (
    [string]$task = "help"
)

function Test {
    Write-Output "Running unittests..."
    python -m unittest discover -s main/unitTest -p "test*.py"
}

function ManualTest {
    Write-Output "Running manual test image pipeline..."
    python main/manualTestImage.py
}

function Gui {
    Write-Output "Launching test GUI..."
    python main/testGui.py
}

function Clean {
    Write-Output "Cleaning up __pycache__ and .pyc files..."
    Get-ChildItem -Recurse -Include "__pycache__", "*.pyc" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

function CleanOutputs {
    Write-Output "Cleaning generated output images..."
    Get-ChildItem -Path main/unitTest/outputs -Include *.png -Recurse | Remove-Item -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path main/gui_outputs -Include *.png -Recurse | Remove-Item -Force -ErrorAction SilentlyContinue
}

function Help {
    Write-Output "Available commands:"
    Write-Output "  ./run.ps1 test           # Run all unit tests"
    Write-Output "  ./run.ps1 manual-test    # Run the manual image test pipeline"
    Write-Output "  ./run.ps1 gui            # Launch test GUI"
    Write-Output "  ./run.ps1 clean          # Remove __pycache__ and .pyc files"
    Write-Output "  ./run.ps1 clean-outputs  # Remove image output files"
}

switch ($task) {
    "test"          { Test }
    "manual-test"   { ManualTest }
    "gui"           { Gui }
    "clean"         { Clean }
    "clean-outputs" { CleanOutputs }
    default         { Help }
}
