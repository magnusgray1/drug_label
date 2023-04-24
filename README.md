This repository contains the code, data, and results for the paper titled "Classifying Free Texts Into Pre-Defined Sections Using AI in Regulatory Documents: A Case Study with Drug Labeling Documents" by Magnus Gray, Joshua Xu, Weida Tong, and Leihong Wu. 


The following describes the files and file structure. 


Folders:
 
	'data': contains the data files used for training and testing the models

		'binary': contains the data files (.csv) for the binary classification task
		'multi': contains the data files (.csv) for the multi-class task

	'results': contains the primary results for the BERT-based models

		'binary': contains the BERT results for the binary classification task

			'eval_logs': contains the testing results (.txt)
			'graphs': contains plots of training loss and accuracy (.png)
			'train_logs': contains the training history logs (.txt)

		'multi': contains the BERT results for the multi-calss task

			'eval_logs': contains the testing results (.txt)
			'graphs': contains plots of training loss and accuracy (.png)
			'train_logs': contains the training history logs (.txt)

	'rf': contains the results for the random forest models

	'shap': contains the results of the SHAP model explainability analysis

	'svm': contains the results for the SVM models
	
	
Python Files:

	'binary.py': the code for fine-tuning and evaluating the binary BERT-based models

	'multi.py': the code for fine-tuning and evaluating the multi-class BERT-based models

	'rf.py': the code for fine-tuning and evaluating the random forest models

	'shap.py': the code for the SHAP model explainability analysis

	'svm.py': the code for fine-tuning and evaluating the SVM models

