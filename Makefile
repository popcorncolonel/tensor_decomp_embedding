all:
	time srun --partition=gpu --time=47:59:00 python3 test_gensim.py
buildvocab:
	time srun --partition=gpu --time=00:30:00 python3 test_gensim.py --buildvocab
interactive:
	time srun --partition=interactive --time=2:00:00 python3 test_gensim.py --load
