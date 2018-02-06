# master_thesis
my master thesis about meta learning for classification task

python3.5/3.6 with tensorflow1.4
dataset: omniglot and miniImagenet

1. train an embedding module first (use resnet and softmax cross_entropy function)
   try auto_encoder as well
2. add pixel-wise loss to avoid overfitting(optional)
3. feature vectors generation(could use the embedding module to train a transformation function to generate feature vectors to add few bias for DeepComparNet) and images augmentation(transforamtion, scale, rotation, crop, whitening)
4. train DeepCompareNet by using the embedding module as parameters initialzation, add comparsion module
   could also train the DeepCompareNet(include embedding module and comparsion moudle from scratch)


details:
1. TODO: choose suitable batch_size and query images size
2. check the gradient
3. check the ratio of updated gradient norm and norm of weight (10^-3, if lower, then learning is slow, if higher, may not stable)
4. if the accuracy on validation dataset is not getting higher any more, then divide the learing rate by 2 or 5
5. ensemble


questions:
1. can capsule network deal with few shots learning?
2. how squared hinge loss works?

just combine images in numpy file


name_scope does not affect for tf.get_variable()  for tf.Variable() with same variable name, var1, var1_1, var1_2
variable_scope has effect for tf.get_variable()

tf.Variable() can not reuse previos defined variable
