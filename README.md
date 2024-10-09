## AILayerImplementations
When i started learning how machine learning and deep learning neurla networks work, i first went into the maths to really see how these systems really do things under the hood. For example how pytorch uses DAGs(Directed Acyclic Graphs) when internally building and managing, computational, graphs inorder to track the sequence of operations during teh forwad pass, allowring for efficent backpropagation during trainig, A CNN's filters and how they are trained to learn different features from an image(and later videos), the cells in RNN and LSTM Layers, how Graph neural networks use an adjancecy matrix, and features of every node plus message passing to learn how different features propagate through each node hence making learning of simulations possible. And of course not forgetting the different stastical machine learning algorithms eg decision trees, random forests, ARIMA, SARIMA models, naive bayes etc.

Yeah so that was a mouthfull but it is the summary of the understanding i have in machine learning and deep learning.

Then after that i started using the apis of tensorflow and pytorch to develop and train different kinds of ai models.

But this project is different from all those, In this am trying to implement the layers themselves depending on how i think they work as explained briely above so to see how these layers that we use in the beautiful apis of tensorflow, pytorch, fastai etc really work.

## caution 
By no cause should anyone try to take these implementations for any production purposes especially when they do not what they are doing, since they are really broken and bad. This is for my own  educational purposes.