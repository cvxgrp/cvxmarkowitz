pipx install poetry
poetry config virtualenvs.in-project true
poetry install
poetry run pip install pre-commit
poetry run pre-commit install
