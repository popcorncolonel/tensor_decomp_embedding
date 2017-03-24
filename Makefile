min_count = 2000
num_sents = 10e6
embedding_dim = 300

cnn:
	python3 -m pdb -c continue test_gensim.py --method=cnn --min_count=$(min_count) --num_sents=$(num_sents) --embedding_dim=$(embedding_dim)
cbow:
	python3 -m pdb -c continue test_gensim.py --method=cbow --min_count=$(min_count) --num_sents=$(num_sents) --embedding_dim=$(embedding_dim)
cp_4d:
	python3 -m pdb -c continue test_gensim.py --method=cp_4d --min_count=$(min_count) --num_sents=$(num_sents) --embedding_dim=$(embedding_dim)
cp_nonneg_4d:
	python3 -m pdb -c continue test_gensim.py --method=cp_nonneg_4d --min_count=$(min_count) --num_sents=$(num_sents) --embedding_dim=$(embedding_dim)
cp_nonneg:
	python3 -m pdb -c continue test_gensim.py --method=cp_nonneg --min_count=$(min_count) --num_sents=$(num_sents) --embedding_dim=$(embedding_dim)
cp_nonneg_poisson:
	python3 -m pdb -c continue test_gensim.py --method=cp_nonneg_poisson --min_count=$(min_count) --num_sents=$(num_sents) --embedding_dim=$(embedding_dim)
cp:
	python3 -m pdb -c continue test_gensim.py --method=cp --min_count=$(min_count) --num_sents=$(num_sents) --embedding_dim=$(embedding_dim)
cp_poisson:
	python3 -m pdb -c continue test_gensim.py --method=cp_poisson --min_count=$(min_count) --num_sents=$(num_sents) --embedding_dim=$(embedding_dim)
nnse:
	python3 -m pdb -c continue test_gensim.py --method=nnse --min_count=$(min_count) --num_sents=$(num_sents) --embedding_dim=$(embedding_dim)
svd:
	python3 -m pdb -c continue test_gensim.py --method=svd --min_count=$(min_count) --num_sents=$(num_sents) --embedding_dim=$(embedding_dim)
random:
	python3 -m pdb -c continue test_gensim.py --method=random --min_count=$(min_count) --num_sents=$(num_sents) --embedding_dim=$(embedding_dim)
sktensor:
	python3 -m pdb -c continue test_gensim.py --method=sktensor --min_count=$(min_count) --num_sents=$(num_sents) --embedding_dim=$(embedding_dim)
restore_ckpt:
	python3 -m pdb -c continue test_gensim.py --method=restore_ckpt --min_count=$(min_count) --num_sents=$(num_sents) --embedding_dim=$(embedding_dim)
loadmatlab:
	python3 -m pdb -c continue test_gensim.py --method=loadmatlab --min_count=$(min_count) --num_sents=$(num_sents) --embedding_dim=$(embedding_dim)
matlab:
	python3 -m pdb -c continue test_gensim.py --method=matlab --min_count=$(min_count) --num_sents=$(num_sents) --embedding_dim=$(embedding_dim)
train_models:
	python3 -m pdb -c continue train_models.py --embedding_dim=$(embedding_dim) --min_count=$(min_count) --num_sents=$(num_sents)

