run:
	python run.py embeddings \
		&& python run.py projection \
		&& tensorboard --logdir='./logs/'

build_embeddings:
	python run.py embeddings

build_projection:
	python run.py projection

run_projector:
	tensorboard --logdir='./logs/'
