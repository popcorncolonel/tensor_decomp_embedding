all:
	time python3 test_gensim.py
buildvocab:
	time python3 test_gensim.py --buildvocab
load_non_interactive:
	time python3 test_gensim.py --load
interactive:
	time python3 test_gensim.py --load
