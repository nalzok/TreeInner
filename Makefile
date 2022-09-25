.PHONY: oracle

oracle:
	time pipenv run python3 -m pyinstrument -m src.main
