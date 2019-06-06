inference:
	# Running decomposition model for bridging
	python3 main.py --do_predict --model span-predictor \
			--output_dir out/decom-bridge \
			--init_checkpoint model/decom-bridge/model.pt \
			--predict_file data/hotpot-all/dev.json,data/decomposition-data/decomposition-bridge-dev-v1.json \
			--max_seq_length 100 --max_n_answers 1 --prefix dev_ --with_key
	# Running decomposition model for intersection
	python3 main.py --do_predict --model span-predictor \
			--output_dir out/decom-intersec \
			--init_checkpoint model/decom-intersec/model.pt \
			--predict_file data/hotpot-all/dev.json,data/decomposition-data/decomposition-intersec-dev-v1.json \
			--max_seq_length 100 --max_n_answers 1 --prefix dev_
	# Preprocessing sub-questions for bridging
	python3 run_decomposition.py --task decompose --data_type dev_b --out_name out/decom-bridge
	# Answering to the first sub-question
	python3 main.py --do_predict --output_dir out/onehop \
			--predict_file data/decomposed/dev_b.1.json \
			--init_checkpoint model/onehop/model1.pt,model/onehop/model2.pt,model/onehop/model3.pt \
			--prefix dev_b_1_ --n_best_size 4
	# Plugging the answer to the second sub-question
	python3 run_decomposition.py --task plug --data_type dev_b --topk 10
	# Answering to the second sub-question
	python3 main.py --do_predict --output_dir out/onehop \
			--predict_file data/decomposed/dev_b.2.json \
			--init_checkpoint model/onehop/model1.pt,model/onehop/model2.pt,model/onehop/model3.pt \
			--prefix dev_b_2_ --n_best_size 4
	# Saving the answer through bridging
	python3 run_decomposition.py --task aggregate-bridge --data_type dev_b --topk 10
	# Preprocessing suub-questions for intersection
	python3 run_decomposition.py --task decompose --data_type dev_i --out_name out/decom-intersec
	# Answering to the first sub-question
	python3 main.py --do_predict --output_dir out/onehop
			--predict_file data/decomposed/dev_i.1.json \
			--init_checkpoint model/onehop/model1.pt,model/onehop/model2.pt,model/onehop/model3.pt \
			--prefix dev_i_1_ --n_best_size 4
	# Answering to the second sub-question
	python3 main.py --do_predict --output_dir out/onehop \
			--predict_file data/decomposed/dev_i.2.json \
			--init_checkpoint model/onehop/model1.pt,model/onehop/model2.pt,model/onehop/model3.pt \
			--prefix dev_i_2_ --n_best_size 4
	# Saving the answer through intersection
	python3 run_decomposition.py --task aggregate-intersec --data_type dev_i --topk 10
	# Answering to the original question
	python3 main.py --do_predict --output_dir out/hotpot \
			--predict_file data/hotpot-all/dev.json \
			--init_checkpoint model/hotpot/model.pt --max_seq_length 300 --prefix dev_ --n_best_size 4
	# Saving the answer through original
	python3 run_decomposition.py --task onehop --data_type dev --topk 10
	# Running the decomposition scorer
	python3 main.py --do_predict --output_dir out/scorer --model classifier \
			--predict_file data/hotpot-all/dev.json,comparison,bridge,intersec,onehop \
			--init_checkpoint model/scorer/best-model.pt \
			--max_seq_length 400 --prefix dev_



