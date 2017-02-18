method = cp_decomp
min_count = 420
num_sents = 5e6

all:
	python3 -m pdb -c continue test_gensim.py
buildvocab:
	python3 -m pdb -c continue test_gensim.py --method=$(method) --min_count=$(min_count) --num_sents=$(num_sents) --buildvocab
cp:
	python3 -m pdb -c continue test_gensim.py --method=$(method) --min_count=$(min_count) --num_sents=$(num_sents)
