model = semantic
target = $(model)-submission.csv

$(target): data/*.csv
	@which train-basic-recommender > /dev/null || pip install -e .
	train-basic-recommender --name $(model) --output $@


develop: data/*.csv
	@which develop-basic-recommender > /dev/null || pip install -e .
	develop-basic-recommender --name $(model)

.PHONY: all
