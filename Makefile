method = matlab
min_count = 800
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
restore_ckpt:
	python3 -m pdb -c continue test_gensim.py --method=restore_ckpt --min_count=$(min_count) --num_sents=$(num_sents)
loadmatlab:
	python3 -m pdb -c continue test_gensim.py --method=loadmatlab --min_count=$(min_count) --num_sents=$(num_sents)
matlab:
	python3 -m pdb -c continue test_gensim.py --method=matlab --min_count=$(min_count) --num_sents=$(num_sents)
