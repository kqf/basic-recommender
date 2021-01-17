model = semantic
target = $(model)-submission.csv

$(target): data/*.csv
	@which train-basic-recommender > /dev/null || pip install -e .
	train-basic-recommender --name $(model) --submission $(target)


develop: data/*.csv
	@which develop-basic-recommender > /dev/null || pip install -e .
	develop-basic-recommender --name $(model)

.PHONY: all
