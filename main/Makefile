# Makefile

.PHONY: test clean manual-test

test:
	python -m unittest discover -s main/unitTest -p "test*.py"

manual-test:
	python main/manualTestImage.py

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	rm -f dummy_test_image.png
