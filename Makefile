.PHONY: oracle

oracle:
	pipenv run python3 -m pyinstrument -m src.oracle
