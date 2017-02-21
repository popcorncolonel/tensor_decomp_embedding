method = cnn
min_count = 420
num_sents = 5e6

all:
	python3 -m pdb -c continue test_gensim.py
buildvocab:
	python3 -m pdb -c continue test_gensim.py --method=$(method) --min_count=$(min_count) --num_sents=$(num_sents) --buildvocab
cnn:
	python3 -m pdb -c continue test_gensim.py --method=cnn --min_count=$(min_count) --num_sents=$(num_sents)
cbow:
	python3 -m pdb -c continue test_gensim.py --method=cbow --min_count=$(min_count) --num_sents=$(num_sents)
cp:
	python3 -m pdb -c continue test_gensim.py --method=cp_decomp --min_count=$(min_count) --num_sents=$(num_sents)
svd:
	python3 -m pdb -c continue test_gensim.py --method=svd --min_count=$(min_count) --num_sents=$(num_sents)
hosvd:
	python3 -m pdb -c continue test_gensim.py --method=hosvd --min_count=$(min_count) --num_sents=$(num_sents)
