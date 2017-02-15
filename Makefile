all:
	python3 -m pdb -c continue test_gensim.py
buildvocab:
	python3 -m pdb -c continue test_gensim.py --buildvocab
cp:
	python3 -m pdb -c continue test_gensim.py --method=cp_decomp --min_count=420 --num_sents=5e6 --buildvocab
load:
	python3 -m pdb -c continue test_gensim.py --method=cp_decomp --min_count=420 --num_sents=5e6
