min_count = 1000
num_articles = 1e5
embedding_dim = 300

random:
	python3 -m pdb -c continue test_gensim.py --method=random --min_count=$(min_count) --num_articles=$(num_articles) --embedding_dim=$(embedding_dim)

cp:
	python3 -m pdb -c continue test_gensim.py --method=cp --min_count=$(min_count) --num_articles=$(num_articles) --embedding_dim=$(embedding_dim)
cp-s_nodebug:
	python3 test_gensim.py --method=cp-s --min_count=$(min_count) --num_articles=$(num_articles) --embedding_dim=$(embedding_dim)
cp-s:
	python3 -m pdb -c continue test_gensim.py --method=cp-s --min_count=$(min_count) --num_articles=$(num_articles) --embedding_dim=$(embedding_dim)
cp-s-gpu:
	python3 -m pdb -c continue test_gensim.py --method=cp-s --min_count=$(min_count) --num_articles=$(num_articles) --embedding_dim=$(embedding_dim) --gpu=True
cp-sn:
	python3 -m pdb -c continue test_gensim.py --method=cp-sn --min_count=$(min_count) --num_articles=$(num_articles) --embedding_dim=$(embedding_dim)
cp-s_4d:
	python3 -m pdb -c continue test_gensim.py --method=cp-s_4d --min_count=$(min_count) --num_articles=$(num_articles) --embedding_dim=$(embedding_dim)
cp-sn_4d:
	python3 -m pdb -c continue test_gensim.py --method=cp-sn_4d --min_count=$(min_count) --num_articles=$(num_articles) --embedding_dim=$(embedding_dim)

jcp-s:
	python3 -m pdb -c continue test_gensim.py --method=jcp-s --min_count=$(min_count) --num_articles=$(num_articles) --embedding_dim=$(embedding_dim)
jcp-s-gpu:
	python3 -m pdb -c continue test_gensim.py --method=jcp-s --min_count=$(min_count) --num_articles=$(num_articles) --embedding_dim=$(embedding_dim) --gpu=True
jcp-s_432:
	python3 -m pdb -c continue test_gensim.py --method=jcp-s_432 --min_count=$(min_count) --num_articles=$(num_articles) --embedding_dim=$(embedding_dim)

cbow:
	python3 -m pdb -c continue test_gensim.py --method=cbow --min_count=$(min_count) --num_articles=$(num_articles) --embedding_dim=$(embedding_dim)
sgns:
	python3 -m pdb -c continue test_gensim.py --method=sgns --min_count=$(min_count) --num_articles=$(num_articles) --embedding_dim=$(embedding_dim)
hosg:
	python3 -m pdb -c continue test_gensim.py --method=hosg --min_count=$(min_count) --num_articles=$(num_articles) --embedding_dim=$(embedding_dim)
svd:
	python3 -m pdb -c continue test_gensim.py --method=svd --min_count=$(min_count) --num_articles=$(num_articles) --embedding_dim=$(embedding_dim)
cnn:
	python3 -m pdb -c continue test_gensim.py --method=cnn --min_count=$(min_count) --num_articles=$(num_articles) --embedding_dim=$(embedding_dim)
nnse:
	python3 -m pdb -c continue test_gensim.py --method=nnse --min_count=$(min_count) --num_articles=$(num_articles) --embedding_dim=$(embedding_dim)
glove:
	python3 -m pdb -c continue test_gensim.py --method=glove --min_count=$(min_count) --num_articles=$(num_articles) --embedding_dim=$(embedding_dim)

restore_ckpt:
	python3 -m pdb -c continue test_gensim.py --method=restore_ckpt --min_count=$(min_count) --num_articles=$(num_articles) --embedding_dim=$(embedding_dim)
matlab:
	python3 -m pdb -c continue test_gensim.py --method=matlab --min_count=$(min_count) --num_articles=$(num_articles) --embedding_dim=$(embedding_dim)
train_models:
	python3 -m pdb -c continue train_models.py --embedding_dim=$(embedding_dim) --min_count=$(min_count) --num_articles=$(num_articles)
